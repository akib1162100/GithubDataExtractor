import os
import git

repo=git.Repo("GithubDataExtractor/")
log=repo.git.log()
# print(log)
commits_list = list(repo.iter_commits())
# diff = repo.git.diff(repo.head.commit.tree)
# print(commits_list)
for commits in commits_list:
    print(commits.stats.total)
    print(commits.stats.files)
    print(commits.message)
    print(commits.committed_datetime)
    print(commits.committer)
# size=diff_size(diff)
# print(commits_list)
# changed_files = []

def diff_size(diff):
    """
    Computes the size of the diff by comparing the size of the blobs.
    """
    if diff.b_blob is None and diff.deleted_file:
        # This is a deletion, so return negative the size of the original.
        return diff.a_blob.size * -1

    if diff.a_blob is None and diff.new_file:
        # This is a new file, so return the size of the new value.
        return diff.b_blob.size

    # Otherwise just return the size a-b
    return diff.a_blob.size - diff.b_blob.size


# for x in commits_list[0].diff(commits_list[-1]):
#     if x.a_blob.path not in changed_files:
#         changed_files.append(x.a_blob.path)
#
#     if x.b_blob is not None and x.b_blob.path not in changed_files:
#         changed_files.append(x.b_blob.path)
diff = commits_list[4].diff(commits_list[0],create_patch=True,ignore_blank_lines=True,ignore_space_at_eol=True,diff_filter='cr')
diff_2=diff[0]
# print(lambda x,y:str(x)+str(y),diff[0])

stri="%s %s"%(lambda x,y:str(x)+str(y),diff[0])
strs=stri.splitlines()
data=strs[4].split()

# hcommit = repo.head.commit.tree
# hcommit.diff()                  # diff tree against index
# print(hcommit.diff())
# hcommit.diff('HEAD~1')          # diff tree against previous tree
# hcommit.diff(None)              # diff tree against working tree
#
# index = repo.index
# index.diff()                    # diff index against itself yielding empty diff
# index.diff(None)                # diff index against working copy
# index.diff('HEAD')
diff_added = []
for diff_add in commits_list[1].diff(commits_list[0]).iter_change_type('A'):
    print("A blob:\n{}".format(diff_add.a_blob.data_stream.read().decode('utf-8')))
    print("B blob:\n{}".format(diff_add.b_blob.data_stream.read().decode('utf-8')))

    diff_added.append(diff_add)

# print(lambda x,y:str(x)+str(y),diff_added[1])
# print(diff_added[1].b_blob.path)