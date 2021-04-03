from django import forms

class RepoNameForm(forms.Form):
    repo_name = forms.CharField(label='Repo name', max_length=100)