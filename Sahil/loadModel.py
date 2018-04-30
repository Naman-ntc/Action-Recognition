import twoLSTMcuda as t
model = t.torch.load(open("twoLSTMentireModel.npy", 'rb'))
print(t.checkAcc(model, t.data, t.labels))
print(t.checkAcc(model, t.valData, t.valLabels))
