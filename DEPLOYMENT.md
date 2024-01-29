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
docker compose restart flowspeech
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
git pull && docker compose restart flowspeech
```

Rebuild and launch right away:

```
docker compose up -d --no-deps --build flowspeech
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

Then `docker compose -f docker-compose_streamlit.yml up -d`. The streamlit app will be reachable locally
as [http://localhost:8001].

# Additional notes:

To kill all containers, e.g. to restart afterwards:

```
docker container kill $(docker ps -q)
```

To remove all images:

```
docker system prune -a
```

If you just want to use it locally, without nginx server, you can build and run only the flowspeech image:

```
docker compose up --build -d flowspeech
```

Connect to a bash terminal without affecting the running state.

```
docker exec -it flaskapp bash
```

To show terminal output:

```
docker logs flowspeech > docker_logs.txt # to get all history, too long if running for a while
docker logs flowspeech --tail=100 # to print last history
docker compose logs --tail=20 --follow # to attach to all containers in docker compose and get what's following. Change tail=10 to have 10 last events
```

# Bootstrap the environment and manage dependencies

The environment is locked to ensure reproducibility. In other to start from scratch using the base `env.yml`, run the following:

```
docker run --rm --user 0 -v "$(pwd):/tmp" \
   mambaorg/micromamba:1.5.6 /bin/bash -c "\
     apt-get update && apt-get install --no-install-recommends -y pipx && \
     pipx run conda-lock -p osx-64 -p linux-64 -f env.yml --without-cuda"
```

The Dockerfile then uses `conda-lock.yml` to install the conda environment.
