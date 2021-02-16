import git
import math
from copy import deepcopy
import pydriller as pyd


def diffusion_metrics_extraction(commit_list):
    list_to_return = []
    for commit in commit_list:
        commit_dict = commit.stats.files
        ns = 0
        nd = 0
        nf = 0
        all_line_modded = 0
        change_in_files = []
        for keys, details in commit_dict.items():
            keys = str(keys)
            change_in_files.append(details['lines'])
            all_line_modded = all_line_modded + details['lines']
            modified_changes = keys.split("/")
            if len(modified_changes) == 1:
                nf = nf+1
            if len(modified_changes) == 2:
                nf = nf+1
                ns = ns+1
            if len(modified_changes) >= 3:
                nf = nf+1
                ns = ns+1
                nd = nd+1
            if ns == 0 or nd == 0:
                ns = 1
                nd = 1

        entropy = 0
        for change_count in change_in_files:
            file_bias =(change_count/all_line_modded)
            entropy = -file_bias* math.log(file_bias,2) + entropy

        diffusion_metrics_dictionary = {'ns': ns,'nd': nd, 'nf': nf, 'entropy': entropy}
        list_to_return.append({'commit': commit.hexsha, 'diffusion_metrics': diffusion_metrics_dictionary})

    return list_to_return


def size_metrics_extraction(commit_list):
    list_to_return=[]
    list_of_files_name = []
    file_line_dict = {}
    commits_list = []
    unmodded_file_line_dict = []
    for commit in commit_list:
        commits_list.append(commit)

    for i in range(len(commits_list)):
        lt = 0
        la = 0
        ld = 0
        commit=commits_list[i]
        this_file_line_dict = {}
        commit_dict = commit.stats.files
        all_line_modded = 0
        change_in_files = []

        for keys, details in commit_dict.items():
            keys = str(keys)
            incertions = details['insertions']
            la = la + incertions
            deletions = details['deletions']
            ld = ld + deletions
            change_in_files.append(details['lines'])
            all_line_modded = all_line_modded + details['lines']
            modified_changes = keys.split("/")
            if list_of_files_name.count(modified_changes[-1]) != 0:
                file_line_dict[modified_changes[-1]] = file_line_dict[modified_changes[-1]]+ incertions-deletions
                this_file_line_dict[modified_changes[-1]] = file_line_dict[modified_changes[-1]]+ incertions- deletions
            else:
                list_of_files_name.append(modified_changes[-1])
                file_line_dict[modified_changes[-1]] = details['lines']
                this_file_line_dict[modified_changes[-1]] = details['lines']

        copy_file_dict = deepcopy(file_line_dict)
        unmodded_file_line_dict.append(copy_file_dict)

        if i == 0:
            lt = 0
            la = 0
            ld = 0
            size_metrics_dict={'lt': lt,'la': la,'ld':ld}
            list_to_return.append({'commit': commit,'size_metrics':size_metrics_dict})
        else:
            lt_dict=unmodded_file_line_dict[i-1]
            for key, value in lt_dict.items():
                if key in this_file_line_dict:
                    lt = lt + value
            if lt ==0:
                la = 0
                ld = 0
                size_metrics_dict = {'lt': lt, 'la': la, 'ld': ld}
                list_to_return.append({'commit': commit, 'size_metrics': size_metrics_dict})
            else:
                la = la/lt
                ld = ld/lt
                size_metrics_dict = {'lt': lt, 'la': la, 'ld': ld}
                list_to_return.append({'commit': commit,'size_metrics':size_metrics_dict})


    return list_to_return



repository_name = "GithubDataExtractor/"
repo = git.Repo(repository_name)
commit_list = list(repo.iter_commits())
length = len(commit_list)
commit_list = reversed(commit_list)
# print(diffusion_metrics_extraction(commit_list))
print(size_metrics_extraction(commit_list))