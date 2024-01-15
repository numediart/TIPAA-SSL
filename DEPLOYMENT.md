# The speech tech

Download code:

```
git clone https://github.com/flowchase/flowspeech
cd flowspeech

# this is for deploying the latest release
# git checkout $(git describe --tags $(git rev-list --tags --max-count=1))

# this is for deploying the release e.g. v1.3.0
# git checkout v1.3.0
```

Build docker containers then run them:

```
docker compose build
docker compose up -d
```

If neither `docker compose` or `docker-compose` work, install it:
https://docs.docker.com/engine/install/debian/

https://computingforgeeks.com/how-to-install-docker-on-debian-12-bookworm/?expand_article=1&expand_article=1

`sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin`

https://docs.docker.com/compose/install/

Restart container:

```
docker compose restart flaskapp
```

or

```
docker compose down
docker compose up -d
```

## Process for updating code on server

- Do a pytest locally
<!-- - docker compose build containers locally
- docker compose up it locally, to check that it runs, and use test_DL_api.py on localhost -->
- ask Filipe to redirect traffic elsewhere to work freely
- On the web server: ssh to it, then, git pull (or git fetch + git merge) and build flaskapp, then up
- check on swagger ui that the default examples work
- run the test_DL_api.py functions from local towards the server

If need to go back before git pull

```
git reset --hard master@{"10 minutes ago"}
```

https://stackoverflow.com/questions/1223354/undo-git-pull-how-to-bring-repos-to-old-state

When you want to update the app with changes in code (no new dependencies):

```
git pull && docker compose restart flaskapp
```

Rebuild and launch right away:

```
docker compose up -d --no-deps --build flaskapp
```

## github actions resources

https://brew.sh/

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

run github actions locally
https://github.com/nektos/act

docker compose pytest actions:
https://github.com/villekr/github-actions-dockercompose-pytest/blob/master/.github/workflows/actions.yaml

# For content-tools

Same for downloading code and models, but run `docker compose -f docker-compose_streamlit.yml build`

Add a `.streamlit/secrets.toml` file containing `password="..."`. The password is stored on 1password.

Then `docker compose -f docker-compose_streamlit.yml up -d`

# Additional notes:

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
docker compose logs --tail=20 --follow # to attach to all containers in docker compose and get what's following. Change tail=10 to have 10 last events
```
