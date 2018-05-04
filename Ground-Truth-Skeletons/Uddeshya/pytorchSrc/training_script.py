from Models import *

################# reading data ######################
dframe1 = pd.read_csv('../csv_data/action_1.csv')
dframe2 = pd.read_csv('../csv_data/action_2.csv')
dframe3 = pd.read_csv('../csv_data/action_3.csv')
dframe4 = pd.read_csv('../csv_data/action_4.csv')
dframe5 = pd.read_csv('../csv_data/action_5.csv')
dframe6 = pd.read_csv('../csv_data/action_6.csv')
dframe7 = pd.read_csv('../csv_data/action_7.csv')
print('read the data, creating data loader ...')
train_split, test_split = train_test_split([dframe1])

SEQ_LEN = None #variable length sequence
ACTd = Data_gen(train_split, SEQ_LEN)
print('dataloader done!')
#####################################################

############### Model instantiation #################
print('instantiating the model ...')
model0 = LSTMClassifierX0(75, 256, 7, 1, 2)
model0 = model0.cuda()
print('model instantiated!')
#####################################################

################ Helper functions ###################
def eval_test_accuracy(model, test_split):
    pred_labels = np.empty(len(test_split))
    orig_labels = np.array([t[1] for t in test_split])
    for i in range(len(test_split)):
        d_in = autograd.Variable(torch.from_numpy(test_split[i][0]).float().cuda())
        d_in = d_in.view(d_in.size()[0], 1, -1)
        y_pred = model(d_in)
        pred_labels[i] = y_pred.data.cpu().max(1)[1].numpy()[0];
    n_samples = len(pred_labels)
    res=(orig_labels==pred_labels)
    correct_count = (res==True).sum()
    return (correct_count*100/n_samples)

def train(model, num_epoch, num_iter, rec_interval, disp_interval):
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    loss_values = []
    avg_loss_values = []
    rec_step = 0
    print('Starting the training ...')
    for eph in range(num_epoch):
        print('epoch {} starting ...'.format(eph))
        avg_loss = 0
        n_samples = 0
        for i in range(num_iter):
            model.hidden = (model.hidden[0].detach(), model.hidden[1].detach())
            model.zero_grad()
            X,Y = next(ACTd)
            n_samples += len(X)
            X = autograd.Variable(torch.from_numpy(X).float().cuda())
            X = X.view(len(X), 1, -1)
            Y = autograd.Variable(torch.LongTensor(np.array([Y])).cuda())

            y_hat = model(X)       
            loss = F.cross_entropy(y_hat, Y)
            avg_loss += loss.data[0]
            
            if i % disp_interval == 0:
                print('epoch: %d iterations: %d loss :%g' % (eph, i, loss.data[0]))
            if rec_step%rec_interval==0:
                loss_values.append(loss.data[0])
            
            loss.backward()     
            optimizer.step()
            rec_step += 1
            
        avg_loss /= n_samples
        avg_loss_values.append(avg_loss)
        #evaluating model accuracy
        acc = eval_test_accuracy(model, test_split)
        print('epoch: {} <====train track===> avg_loss: {}, test_accuracy: {}% \n'.format(eph, avg_loss, acc))
    return loss_values, avg_loss_values
#####################################################

################## running ##########################
loss_vals, avg_loss_vals = train(model0, 1, 7, 1, 1)
#####################################################

