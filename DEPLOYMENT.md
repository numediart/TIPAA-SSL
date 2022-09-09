
Download codes and models:
```
git clone https://github.com/flowchase/flowspeech
cd flowspeech

# this is for deploying the latest release
# git checkout $(git describe --tags $(git rev-list --tags --max-count=1))

# this is for deploying the release e.g. v1.3.0
# git checkout v1.3.0

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

If neither `docker compose` or `docker-compose` work, install it: 
https://docs.docker.com/engine/install/debian/

`sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin`

https://docs.docker.com/compose/install/

Restart container:
```
docker-compose restart flaskapp
```

or
```
docker-compose down
docker-compose up -d
```

When you want to update the app with changes in code (no new depencies):
```
git pull && docker compose restart flaskapp
```

Rebuild and launch right away:
```
docker-compose up -d --no-deps --build flaskapp
```


Additional notes: 

To kill all containers, e.g. to restart afterwards:
```
docker container kill $(docker ps -q)
```

To remove all images:
```
docker system prune -a
```


If you just want to use it locally, without nginx server, you can build only flowspeech image:
```
docker build -t flowspeech .
docker run -d -p 8000:8000 flowspeech
```

Connect to a bash terminal without affecting the running state. 
```
docker exec -it flaskapp bash
```

To show terminal output:
```
docker logs flaskapp > docker_logs.txt # to get all history, too long if running for a while
docker logs flaskapp --tail=100 # to print last history
docker-compose logs --tail=20 --follow # to attach to all containers in docker compose and get what's following. Change tail=10 to have 10 last events
```

