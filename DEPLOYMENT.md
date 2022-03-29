
Download codes and models:
```
git clone https://github.com/flowchase/flowspeech
cd flowspeech
git clone https://github.com/noetits/charsiu
sudo apt-get install git-lfs
git lfs install
python scripts/download_models.py
```

Build docker containers then run them:
```
docker-compose build
docker-compose up -d
```

Restart container:
```
docker-compose restart flaskapp
```

or
```
docker-compose down
docker-compose up -d
```

Rebuild and launch right away:
```
docker-compose up -d --no-deps --build flaskapp
```
