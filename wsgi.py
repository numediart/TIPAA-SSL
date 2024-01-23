from app import init_app

app = init_app()
debug = True

if __name__ == "__main__":
    app.run(debug=debug, host="0.0.0.0", port=8001)
