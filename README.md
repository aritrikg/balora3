# BALoRA Streamlit Cloud

## Fix the balora2 deployment

1. Extract this ZIP on your computer.
2. Copy the entire `BALoRA_Streamlit` folder into the **root of the balora2 GitHub repository**. If the folder already exists, merge it and replace the included files. Commit the extracted files to `main` with their original folder structure.
3. In GitHub, check that both `BALoRA_Streamlit/study_data.py` and `BALoRA_Streamlit/data/per_prompt.csv.gz` are present on `main`.
4. Keep the Streamlit entry point at `BALoRA_Streamlit/app.py`. After the new commit is deployed, open https://baloraee.streamlit.app/ and select **Upload CT**.

```text
balora2/
└── BALoRA_Streamlit/
    ├── app.py
    ├── study_data.py
    ├── cloud_upload.py
    ├── upload_scans.py
    ├── requirements.txt
    └── data/
        └── per_prompt.csv.gz
```

The archive includes the additional summaries and saved examples used by the dashboard. For metrics-only public hosting, you can omit `BALoRA_Streamlit/data/examples/`.

## Connect mask generation

The deployed frontend accepts and previews de-identified NIfTI, PNG, and JPEG files. **Generate mask** becomes available after you connect the GPU service from the earlier BALoRA cloud upload package. On Streamlit Cloud, enter your actual service address and token under **App settings → Secrets**.

```toml
[balora_gpu]
url = "https://your-gpu-service.example"
token = "your-generated-service-token"
```

Confirm the service works with a de-identified research image. The dashboard displays the predicted mask and provides a PNG download at the selected slice's original pixel dimensions.

[BALoRA cloud upload package with GPU backend](https://chatgpt.com/library?file_id=libfile_9e8e34ee2e948191b2ad72c65f2247b0&account_id=personal)
