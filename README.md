# AI-Business-Advisor

This project is organized into the following layers:

- `data` for datasets and ETL assets
- `model` for AI/ML models and training artifacts
- `backend` for API and business logic
- `frontend` for web UI
- `mobile` for mobile app code
- `docs` for product and technical documentation

## Structure

```text
AI-Business-Advisor/
├── data/
├── model/
├── backend/
├── frontend/
├── mobile/
├── docs/
└── README.md
```

## Run the application

On Windows, run `START.bat` from the project directory. It starts the FastAPI
backend at `http://127.0.0.1:8000` and the Vite frontend at
`http://127.0.0.1:5173`. Open the frontend URL in a browser.

The frontend sends API requests to `/api`; Vite proxies those requests to the
backend during development. To use a different backend address, set
`API_PROXY_TARGET` before starting Vite. For a frontend deployment without the
Vite development proxy, set `VITE_API_URL` to the API base URL when building
the frontend.
