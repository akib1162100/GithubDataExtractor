import git
import os
repo =git.Repo("analytics/")
commit=repo.head.commit
log=repo.git.log()
