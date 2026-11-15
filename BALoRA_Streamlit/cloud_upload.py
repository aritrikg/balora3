"""Cloud upload and HTTPS client for the private model service."""
from hashlib import sha256
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from PIL import Image
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from upload_scans import MAX_SIDE, png_mask, preview, read_raster, read_volume, window

MODELS = {
    "balora": "Gaussian Decoder LoRA (BALoRA)",
    "deterministic": "Deterministic LoRA",
    "frozen": "Frozen SAM2",
}
MAX_RESPONSE = 8 * 1024**2


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def backend_settings():
    try:
        settings = st.secrets.get("balora_gpu", {})
        endpoint = str(settings.get("url", "")).strip().rstrip("/")
        token = str(settings.get("token", ""))
    except StreamlitSecretNotFoundError:
        return None
    if not endpoint or len(token) < 32:
        return None
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None
    return endpoint, token


def remote_mask(settings, gray, method, point):
    endpoint, token = settings
    h, w = gray.shape
    x, y = point
    query = urlencode({"method": method, "width": w, "height": h, "x": x, "y": y})
    request = Request(f"{endpoint}/segment?{query}", data=gray.astype("<f4", copy=False).tobytes(),
                      headers={"Content-Type": "application/octet-stream", "Authorization": f"Bearer {token}"},
                      method="POST")
    try:
        with build_opener(NoRedirect()).open(request, timeout=120) as response:
            if response.headers.get_content_type() != "image/png":
                raise ValueError("The inference server returned an unexpected response.")
            data = response.read(MAX_RESPONSE + 1)
    except HTTPError as exc:
        messages = {
            401: "The inference server rejected the app credentials. Check the matching access tokens.",
            429: "The GPU server is busy. Try again shortly.",
            503: "The GPU model is unavailable. Check the backend setup or try again shortly.",
        }
        raise ValueError(messages.get(exc.code, f"The inference server returned HTTP {exc.code}.")) from exc
    except (OSError, URLError) as exc:
        raise ValueError("The app could not reach the GPU server over HTTPS.") from exc
    if len(data) > MAX_RESPONSE:
        raise ValueError("The inference server returned an oversized response.")
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (w, h):
            raise ValueError("The GPU server returned a mask on a different pixel grid.")
        return np.asarray(image.convert("L")) > 0


def upload_page():
    st.header("Upload CT")
    st.write("Use a de-identified research CT volume or slice. Choose a point inside the target organ to generate a 2D mask.")
    settings = backend_settings()
    if settings is None:
        st.info("Preview is available. An app administrator must connect the GPU backend to enable mask generation.")
    else:
        st.caption("Streamlit Cloud receives the upload. Generate mask sends the selected slice and prompt to the configured GPU server.")
    upload = st.file_uploader("CT image", type=["nii", "gz", "png", "jpg", "jpeg"], max_upload_size=32,
                              help="NIfTI (.nii or .nii.gz) or already-windowed 8-bit PNG/JPEG.")
    if upload is None:
        st.session_state.pop("cloud_volume", None)
        st.session_state.pop("cloud_result", None)
        return

    raw = upload.getvalue()
    upload_id = sha256(raw).hexdigest()
    filename = upload.name.lower()
    index = 0
    try:
        if filename.endswith((".nii", ".nii.gz")):
            cached = st.session_state.get("cloud_volume")
            if cached is None or cached[0] != upload_id:
                st.session_state.pop("cloud_volume", None)
                st.session_state["cloud_volume"] = (upload_id, read_volume(raw, filename.endswith(".gz")))
            volume = st.session_state["cloud_volume"][1]
            if len(volume.shape) == 3 and volume.shape[2] > 1:
                index = st.slider("Slice along stored axis 3", 0, volume.shape[2] - 1,
                                  volume.shape[2] // 2, key=f"cloud_slice_{upload_id}")
            gray = window(volume.dataobj[:, :, index] if len(volume.shape) == 3 else volume.dataobj)
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            st.session_state.pop("cloud_volume", None)
            gray = read_raster(raw)
        else:
            raise ValueError("Choose a supported CT image.")
        if min(gray.shape) < 2 or max(gray.shape) > MAX_SIDE:
            raise ValueError(f"Use a slice between 2 and {MAX_SIDE} pixels per side.")
    except Exception as exc:
        st.error(f"Could not open the CT image. {exc}")
        return

    h, w = gray.shape
    a, b, c = st.columns(3)
    model = a.selectbox("Model", list(MODELS), format_func=MODELS.__getitem__)
    x = b.number_input("X · column from left", 0, w - 1, w // 2, key=f"cloud_x_{upload_id}")
    y = c.number_input("Y · row from top", 0, h - 1, h // 2, key=f"cloud_y_{upload_id}")
    point = (int(x), int(y))
    key = (upload_id, index, model, point)
    left, right = st.columns(2)
    left.image(preview(gray, point), caption=f"CT slice · positive point {point}", width="stretch")
    if right.button("Generate mask", type="primary", disabled=settings is None):
        st.session_state.pop("cloud_result", None)
        try:
            with st.spinner("Generating a mask on the GPU server"):
                st.session_state["cloud_result"] = (key, remote_mask(settings, gray, model, point))
        except (OSError, ValueError) as exc:
            right.error(str(exc))
    result = st.session_state.get("cloud_result")
    if result is not None and result[0] == key:
        mask = result[1]
        right.image(preview(gray, point, mask), caption="Predicted mask overlay", width="stretch")
        right.download_button("Download binary mask (PNG)", png_mask(mask), f"balora_mask_slice_{index}.png",
                              "image/png", on_click="ignore")
