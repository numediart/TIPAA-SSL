git fetch && git log --pretty=format:"%h %ad%x09%an%x09%s" --name-only --date=short HEAD..FETCH_HEAD>gitlog_from_main.txt
git log --pretty=format:"%h %ad%x09%an%x09%s" --name-only --date=short origin/main..HEAD>gitlog_local_commits.txt
