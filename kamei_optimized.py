import git
import math, collections
import pandas as pd
import numpy as np
from git.repo.base import Repo
from copy import deepcopy
from dateutil.relativedelta import relativedelta
from functools import reduce


def file_related_information(commit_list):
    file_name_list = []
    for commit in commit_list:
        for key, details in commit.stats.files.items():
            file_name_list.append(key)
    file_name_list = list(set(file_name_list))
    file_touched_dict_list = []
    file_touched_commit_list = []
    for file in file_name_list:
        committer_list = []
        commits_list = []
        for commits in commit_list:
            for key in commits.stats.files:
                if file == key:
                    committer_list.append(commits.committer.name)
                    commits_list.append(
                        {'committed_datetime': commits.committed_datetime.date(), 'hexsha': commits.hexsha})

        file_touched_dict_list.append({'file': file, 'committer_name': committer_list})
        file_touched_commit_list.append({'file': file, 'commit': commits_list})
    return (file_touched_dict_list, file_touched_commit_list)


def history_dimention_metrics_extraction(commit_list):
    list_to_return = []
    for i in range(len(commit_list)):
        ndev_list = np.array([])
        nuc_list = np.array([])
        age = 0
        prev_commitlist = commit_list[:i]
        file_touched_dict_list, file_touched_commit_list = file_related_information(prev_commitlist)
        commit = commit_list[i]
        this_files = []
        for key in commit.stats.files:
            this_files.append(key)

        for this_dict in file_touched_dict_list:
            if this_dict['file'] in this_files:
                ndev_list = np.append(ndev_list, this_dict['committer_name'])
        for this_commit_dict in file_touched_commit_list:
            if this_commit_dict['file'] in this_files:
                age = age + (commit.committed_datetime.date() - this_commit_dict['commit'][-1][
                    'committed_datetime']).days
                for commit_sha in this_commit_dict['commit']:
                    nuc_list = np.append(nuc_list, commit_sha['hexsha'])
        age = age / len(this_files)
        ndev = np.unique(ndev_list).size
        nuc = np.unique(nuc_list).size
        list_to_return.append({'commit': commit.hexsha, 'ndev': ndev, 'pd': nuc, 'npt': age})
    df = pd.DataFrame(list_to_return)
    del list_to_return[:]
    return df


def experiance_metrics_extraction(commit_list):
    list_to_return = []
    for i in range(commit_list.size):
        hexsha = commit_list[i].hexsha
        dev_name = commit_list[i].committer.name
        devs_list = []
        devs_rexp_list = []
        prev_commit_list = commit_list[:i + 1]
        number_of_years = relativedelta(commit_list[i].committed_datetime, prev_commit_list[0].committed_datetime).years
        for commit in prev_commit_list:
            devs_list.append(commit.committer.name)
        for j in range(number_of_years + 1):
            committer_list = []
            for commit in prev_commit_list:
                committer_list.append(commit.committer.name)
            devs_rexp_list.append(j)
            devs_rexp_list.append(committer_list)

        devs_exp = {i: devs_list.count(i) for i in devs_list}

        file_touched_dict_list, file_touched_commit_list = file_related_information(prev_commit_list)
        del file_touched_commit_list[:]
        subsystem_dev_list = []
        sub_dict_list = np.array([])
        sub_sys_name_set = set()
        for file_info in file_touched_dict_list:
            if len(file_info['file'].split('/')) >= 2:
                subsystem_dev_list = file_info['committer_name']
                sub_dict_list = np.append(sub_dict_list, {'sub_sys': file_info['file'].split('/')[0],
                                                          'subsys_devs': subsystem_dev_list})
                sub_sys_name_set.add(file_info['file'].split('/')[0])
        aggrigated_sub_dict_list = np.array([])
        for subsysname in sub_sys_name_set:
            aggrigated_dev_list = []
            for dictionary in sub_dict_list:
                if dictionary['sub_sys'] == subsysname:
                    aggrigated_dev_list.extend(dictionary['subsys_devs'])
            aggrigated_sub_dict_list = np.append(aggrigated_sub_dict_list,
                                                 {'sub_sys': subsysname, 'devs': aggrigated_dev_list})

        rexp = 0
        sexp = 0
        subsysname = None
        for key in commit_list[i].stats.files:
            if len(key.split('/')) >= 2:
                subsysname = key.split('/')[0]
                for dictionary in aggrigated_sub_dict_list:
                    if dictionary['sub_sys'] == subsysname:
                        sexp = sexp + dictionary['devs'].count(dev_name)

        for i in range(int(len(devs_rexp_list) / 2)):
            j = (i * 2)
            rexp = rexp + devs_rexp_list[j + 1].count(dev_name) / (devs_rexp_list[j] + 1)

        exp = devs_exp[str(dev_name)]
        list_to_return.append({'commit': hexsha, 'exp': exp, 'rexp': rexp, 'sexp': sexp})
    df = pd.DataFrame(list_to_return)
    del list_to_return[:]
    return df


def diffusion_metrics_extraction(commit_list):
    diffusion_list_to_return = []
    purpose = 0
    for commit in commit_list:
        commit_dict = commit.stats.files
        ns = 0
        nd = 0
        nf = 0
        all_line_modded = 0
        change_in_files = np.array([])

        for keys, details in commit_dict.items():
            keys = str(keys)
            change_in_files = np.append(change_in_files, details['lines'])
            all_line_modded = all_line_modded + details['lines']
            modified_changes = keys.split("/")
            if len(modified_changes) == 1:
                nf = nf + 1
            if len(modified_changes) == 2:
                nf = nf + 1
                ns = ns + 1
            if len(modified_changes) >= 3:
                nf = nf + 1
                ns = ns + 1
                nd = nd + 1
            if ns == 0 or nd == 0:
                ns = 1
                nd = 1
        entropy = 0
        for change_count in change_in_files:
            file_bias = (change_count / all_line_modded)
            entropy = -file_bias * math.log(file_bias, 2) + entropy

        diffusion_list_to_return.append(
            {'commit': commit.hexsha, 'ns': nd, 'nm': ns, 'nf': nf, 'entropy': entropy})
        np.delete(change_in_files, np.s_[:], 0)
    df = pd.DataFrame(diffusion_list_to_return)
    return df


def size_metrics_extraction(commit_list):
    size_list_to_return = []
    list_of_files_name = []
    file_line_dict = {}
    unmodded_file_line_dict = np.array([])
    fix = 0
    for i in range(len(commit_list)):
        lt = 0
        la = 0
        ld = 0
        commit = commit_list[i]
        this_file_line_dict = {}
        commit_dict = commit.stats.files
        all_line_modded = 0
        if 'bug' in commit.message or 'defect' in commit.message or 'fix' in commit.message or 'patch' in commit.message:
            fix = 1
        for keys, details in commit_dict.items():
            keys = str(keys)
            insertions = details['insertions']
            la = la + insertions
            deletions = details['deletions']
            ld = ld + deletions
            all_line_modded = all_line_modded + details['lines']
            modified_changes = keys.split("/")
            if list_of_files_name.count(modified_changes[-1]) != 0:
                file_line_dict[modified_changes[-1]] = file_line_dict[modified_changes[-1]] + insertions - deletions
                this_file_line_dict[modified_changes[-1]] = file_line_dict[
                                                                modified_changes[-1]] + insertions - deletions
            else:
                list_of_files_name.append(modified_changes[-1])
                file_line_dict[modified_changes[-1]] = details['lines']
                this_file_line_dict[modified_changes[-1]] = details['lines']

        copy_file_dict = deepcopy(file_line_dict)
        unmodded_file_line_dict = np.append(unmodded_file_line_dict, copy_file_dict)

        if i == 0:
            lt = 0
            la = 0
            ld = 0
            size_list_to_return.append({'commit': commit.hexsha, 'la': la, 'ld': ld, 'lt': lt, 'fix': fix})
        else:
            lt_dict = unmodded_file_line_dict[i - 1]
            for key, value in lt_dict.items():
                if key in this_file_line_dict:
                    lt = lt + value
            if lt == 0:
                la = 0
                ld = 0
                size_list_to_return.append({'commit': commit.hexsha, 'la': la, 'ld': ld, 'lt': lt, 'fix': fix})
            else:
                la = la / lt
                ld = ld / lt
                size_list_to_return.append({'commit': commit.hexsha, 'la': la, 'ld': ld, 'lt': lt, 'fix': fix})
    del list_of_files_name[:]
    np.delete(unmodded_file_line_dict, np.s_[:], 0)
    df = pd.DataFrame(size_list_to_return)
    del size_list_to_return[:]
    return df


class Stats(object):
    files = None

    def __init__(self, files):
        self.files = files


class Committer(object):
    name = None

    def __init__(self,name):
        self.name = name


class Reduced_commit(object):
    stats = Stats(None)
    committed_datetime = None
    hexsha = None
    message = None
    committer = Committer(None)

    def __init__(self, stats, committed_datetime, hexsha, message, committer):
        self.stats = Stats(stats.files)
        self.committed_datetime = committed_datetime
        self.hexsha = hexsha
        self.message = message
        self.committer = Committer(committer.name)


def main():
    repository_name = "cloned/"
    url = 'https://github.com/akib1162100/GithubDataExtractor'
    # git.Git('/cloned/repo').clone('https://github.com/akib1162100/GithubDataExtractor')
    Repo.clone_from(url,repository_name)
    repo = git.Repo(repository_name)
    commits_list = list(repo.iter_commits())
    reduced_commit_list = []

    for commit in commits_list:
        reduced_commit = Reduced_commit(commit.stats, commit.committed_datetime, commit.hexsha, commit.message, commit.committer )
        reduced_commit_list.append(reduced_commit)

    reduced_commit_list.sort(key=lambda x: x.committed_datetime, reverse=False)
    commit_list = np.array(reduced_commit_list)
    diffusion_metrics = diffusion_metrics_extraction(commit_list)
    size_metrics = size_metrics_extraction(commit_list)
    history = history_dimention_metrics_extraction(commit_list)
    exp = experiance_metrics_extraction(commit_list)
    dfs = [diffusion_metrics, size_metrics, history, exp]
    df_final = reduce(lambda left, right: pd.merge(left, right, on='commit' ), dfs)
    df_final.set_index('commit', inplace=True)
    print(df_final)
    df_final.to_csv(r'preprocessed_data.csv', header=True)


if __name__ == "__main__":
    main()
# TODO : history and experiance metrics and integration
