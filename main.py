from app.main import start_server

if __name__ == "__main__":
    # Start with development settings
    start_server(
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info"
    )