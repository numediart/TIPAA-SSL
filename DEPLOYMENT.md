
Download codes and models:
```
git clone https://github.com/flowchase/flowspeech


cd flowspeech

# this is for deploying a branch called e.g. w2v_dev
# git checkout w2v_dev
# this is for deploying the latest release
# git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
# this is for deploying the release e.g. v1.3.0
git checkout v1.3.0

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

If neither `docker compose` or `docker-compose` work, install it: https://docs.docker.com/compose/install/


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
