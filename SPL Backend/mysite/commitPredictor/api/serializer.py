from rest_framework import serializers

class CommitPrediction(serializers.Serializer):
    commit = serializers.CharField()
    prob = serializers.FloatField()

class CommitDetail(serializers.Serializer):
    commit = serializers.CharField()
    nd = serializers.IntegerField()
    ns = serializers.IntegerField()
    nf = serializers.IntegerField()
    entropy = serializers.FloatField()
    la = serializers.FloatField()
    ld = serializers.FloatField()
    lt = serializers.IntegerField()
    fix = serializers.IntegerField()
    ndev = serializers.IntegerField()
    nuc = serializers.IntegerField()
    age = serializers.FloatField()
    exp = serializers.IntegerField()
    rexp = serializers.FloatField()
    sexp = serializers.IntegerField()
