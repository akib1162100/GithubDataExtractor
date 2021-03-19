import git
import math
from copy import deepcopy
from dateutil.relativedelta import relativedelta


def diffusion_metrics_extraction(commit_list):
    diffusion_list_to_return = []
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

        # diffusion_metrics_dictionary = {'ns': ns,'nd': nd, 'nf': nf, 'entropy': entropy}
        diffusion_list_to_return.append({'commit': commit.hexsha, 'ns': ns,'nd': nd, 'nf': nf, 'entropy': entropy})

    return diffusion_list_to_return


def size_metrics_extraction(commit_list):
    size_list_to_return=[]
    list_of_files_name = []
    file_line_dict = {}
    unmodded_file_line_dict = []

    for i in range(len(commit_list)):
        lt = 0
        la = 0
        ld = 0
        commit=commit_list[i]
        this_file_line_dict = {}
        commit_dict = commit.stats.files
        all_line_modded = 0
        change_in_files = []

        for keys, details in commit_dict.items():
            keys = str(keys)
            insertions = details['insertions']
            la = la + insertions
            deletions = details['deletions']
            ld = ld + deletions
            change_in_files.append(details['lines'])
            all_line_modded = all_line_modded + details['lines']
            modified_changes = keys.split("/")
            if list_of_files_name.count(modified_changes[-1]) != 0:
                file_line_dict[modified_changes[-1]] = file_line_dict[modified_changes[-1]]+ insertions-deletions
                this_file_line_dict[modified_changes[-1]] = file_line_dict[modified_changes[-1]]+ insertions- deletions
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
            # size_metrics_dict={'lt': lt,'la': la,'ld':ld}
            size_list_to_return.append({'commit': commit.hexsha,'lt': lt,'la': la,'ld':ld})
        else:
            lt_dict=unmodded_file_line_dict[i-1]
            for key, value in lt_dict.items():
                if key in this_file_line_dict:
                    lt = lt + value
            if lt == 0:
                la = 0
                ld = 0
                # size_metrics_dict = {'lt': lt, 'la': la, 'ld': ld}
                size_list_to_return.append({'commit': commit.hexsha, 'lt': lt, 'la': la, 'ld': ld})
            else:
                la = la/lt
                ld = ld/lt
                # size_metrics_dict = {'lt': lt, 'la': la, 'ld': ld}
                size_list_to_return.append({'commit': commit.hexsha,'lt': lt, 'la': la, 'ld': ld})


    return size_list_to_return

def purpose_metrics_extraction(commit_list):
    purpose_list_to_return = []
    purpose = 0
    for commit in commit_list:
        if 'bug' in commit.message or 'defect' in commit.message or 'fix' in commit.message or 'patch' in commit.message :
            purpose = 1
        purpose_metrics_dict = {'commit': commit.hexsha,'purpose':purpose}
        purpose_list_to_return.append(purpose_metrics_dict)
    return purpose_list_to_return



def file_related_information(commit_list):
    file_name_list = []
    for commit in commit_list:
        for key, details in commit.stats.files.items():
            file_name_list.append(key)
    file_name_list = list(set(file_name_list))
    file_touched_dict_list = []
    file_touched_commit_list = []
    this_commit_ndev = []
    for file in file_name_list:
        committer_list = []
        commits_list = []
        for commits in commit_list:
            this_commit_ndev.extend(committer_list)
            for key in commits.stats.files:
                if file == key:
                    committer_list.append(commits.committer.name)
                    commits_list.append(commits)

        commits_list.sort(key=lambda x: x.committed_datetime, reverse=False)

        file_touched_dict_list.append({'file': file, 'committer_name': committer_list})
        file_touched_commit_list.append({'file': file, 'commit': commits_list})

    return (file_touched_dict_list , file_touched_commit_list)


def history_dimention_metrics_extraction(commit_list):
    list_to_return = []
    for i in range(len(commit_list)):
        ndev_list = []
        nuc_list = []
        age = 0
        prev_commitlist = commit_list[ :i]
        file_touched_dict_list, file_touched_commit_list = file_related_information(prev_commitlist)
        commit = commit_list[i]
        this_files = []
        for key in commit.stats.files:
            this_files.append(key)

        for this_dict in file_touched_dict_list:
            if this_dict['file'] in this_files:
                ndev_list.extend(this_dict['committer_name'])
        for this_commit_dict in file_touched_commit_list:
            if this_commit_dict['file'] in this_files:
                age =age + (commit.committed_datetime.date()-this_commit_dict['commit'][-1].committed_datetime.date()).days
                for commit in this_commit_dict['commit']:
                    nuc_list.append(commit.hexsha)
        age = age/len(this_files)
        ndev = len(list(set(ndev_list)))
        nuc = len(list(set(nuc_list)))
        list_to_return.append({'commit': commit.hexsha, 'age':age ,'ndev':ndev,'nuc':nuc})
    return list_to_return


def experiance_metrics_extraction(commit_list):
    list_to_return = []
    for i in range(len(commit_list)):
        devs_exp = {}
        devs_list = []
        devs_rexp_list = []
        prev_commit_list = commit_list[:i+1]
        number_of_years = relativedelta(commit_list[i].committed_datetime, prev_commit_list[0].committed_datetime).years
        for commit in prev_commit_list:
            devs_list.append(commit.committer.name)
        for j in range(number_of_years+1):
            committer_list = []
            for commit in prev_commit_list:
                committer_list.append(commit.committer.name)
            devs_rexp_list.append(j)
            devs_rexp_list.append(committer_list)
        devs_exp = {i: devs_list.count(i) for i in devs_list}
        file_touched_dict_list, file_touched_commit_list = file_related_information(prev_commit_list)
        subsystem_dev_list = []
        sub_dict_list=[]
        sub_sys_name_set = set()
        for file_info in file_touched_dict_list:
            if len(file_info['file'].split('/')) >= 2:
                subsystem_dev_list = file_info['committer_name']
                sub_dict_list.append({'sub_sys':file_info['file'].split('/')[0],'subsys_devs':subsystem_dev_list})
                sub_sys_name_set.add(file_info['file'].split('/')[0])
        aggrigated_sub_dict_list = []
        for subsysname in sub_sys_name_set:
            aggrigated_dev_list = []
            for dictionary in sub_dict_list:
                if dictionary['sub_sys'] == subsysname:
                    aggrigated_dev_list.extend(dictionary['subsys_devs'])
            aggrigated_sub_dict_list.append({'sub_sys': subsysname, 'devs': aggrigated_dev_list})


        exp = 0
        rexp = 0
        sexp = 0
        subsysname = None
        for key in commit_list[i].stats.files:
            if len(key.split('/')) >=2:
                subsysname = key.split('/')[0]
                for dictionary in aggrigated_sub_dict_list:
                    if dictionary['sub_sys'] == subsysname:
                        sexp = sexp + dictionary['devs'].count(commit_list[i].committer.name)

        for i in range(int(len(devs_rexp_list)/2)):
            j = (i*2)
            rexp = rexp + devs_rexp_list[j+1].count(commit_list[i].committer.name)/(devs_rexp_list[j]+1)

        exp = devs_exp[str(commit_list[i].committer.name)]
        list_to_return.append({'commit': commit_list[i].hexsha, 'exp': exp, 'rexp': rexp, 'sexp': sexp})
    return list_to_return





def main():
    repository_name = "GithubDataExtractor/"
    repo = git.Repo(repository_name)
    commit_list = list(repo.iter_commits())
    commit_list.sort(key=lambda x: x.committed_datetime, reverse=False)
    diffusion_metrics = diffusion_metrics_extraction(commit_list)
    size_metrics = size_metrics_extraction(commit_list)
    purpose_metrics = purpose_metrics_extraction(commit_list)
    history = history_dimention_metrics_extraction(commit_list)
    exp = experiance_metrics_extraction(commit_list)
    extracted_list = []
    for i in range(len(purpose_metrics)):
        dictionary = {}
        dictionary.update(diffusion_metrics[i])
        dictionary.update(size_metrics[i])
        dictionary.update(purpose_metrics[i])
        dictionary.update(history[i])
        dictionary.update(exp[i])
        extracted_list.append(dictionary)

    print(extracted_list)


if __name__ == "__main__":
    main()
# TODO : history and experiance metrics and integration