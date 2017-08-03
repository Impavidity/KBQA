
from args import get_args
import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from simple_qa_ner import SimpleQADataset
from model import EntityDetection
import time
import os

# please set the configuration in the file : args.py
args = get_args()
# set the random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for training")


# load data with torchtext

questions = data.Field(lower=True, sequential=True)
labels = data.Field(sequential=True)

train, dev, test = SimpleQADataset.splits(questions, labels)

# build vocab for questions
questions.build_vocab(train, dev) # Test dataset can not be used here for constructing the vocab
# build vocab for tags
labels.build_vocab(train, dev)


if os.path.isfile(args.vector_cache):
    questions.vocab.vectors = torch.load(args.vector_cache)
else:
    questions.vocab.load_vectors(wv_dir=args.data_cache, wv_type=args.word_vectors, wv_dim=args.d_embed)
    os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
    torch.save(questions.vocab.vectors, args.vector_cache)

#print(labels.vocab.stoi)
# Buckets
train_iters, dev_iters, test_iters = data.BucketIterator.splits(
    (train, dev, test), batch_size=args.batch_size, device=args.gpu)


train_iters.repeat = False

# define models

config = args
config.n_embed = len(questions.vocab)
config.n_out = len(labels.vocab) # I/in entity  O/out of entity
config.n_cells = config.n_layers

if config.birnn:
    config.n_cells *= 2
print(config)

if args.resume_snapshot:
    pass
else:
    model = EntityDetection(config)
    #if args.word_vectors:
    #    model.embed.weight.data = questions.vocab.vectors
    if args.cuda:
        model.cuda()
        print("Shift model to GPU")

criterion = nn.NLLLoss() # negative log likelyhood loss function
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# train the model
iterations = 0
start = time.time()
best_dev_acc = 0
num_iters_in_epoch = (len(train) // args.batch_size) + 1
patience = args.patience * num_iters_in_epoch # for early stopping
iters_not_improved = 0 # this parameter is used for stopping early
early_stop = False


print("Start to train")

for epoch in range(args.epochs):
    if early_stop:
        print("Early stopping. Epoch: {}. Dest Dev. Acc. : {}".format(epoch, best_dev_acc))
    train_iters.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iters):
        iterations += 1
        model.train()
        optimizer.zero_grad()

        answer = model(batch)
        n_correct += (((torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum(dim=0)) == batch.label.size()[0]).sum()
        #n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        #n_total +=  batch.label.size()[1] * batch.label.size()[0]
        loss = criterion(answer, batch.label.view(-1,1)[:,0])
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
        optimizer.step()

        if iterations % args.log_every == 0:
            print(epoch, n_correct / n_total * 100, loss.data[0])

        if iterations % args.dev_every == 0:
            model.eval()
            dev_iters.init_epoch()
            n_dev_correct = 0
            n_dev_total = 0
            for dev_batch_idx, dev_batch in enumerate(dev_iters):
                answer = model(dev_batch)
                n_dev_correct += (((torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum(dim=0)) == dev_batch.label.size()[0]).sum()
                #n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                #n_dev_total += dev_batch.label.size()[1] * dev_batch.label.size()[0]
                dev_loss = criterion(answer, dev_batch.label.view(-1,1)[:,0])
            dev_acc = 100. * n_dev_correct / len(dev)
            #dev_acc = 100. * n_dev_correct / n_dev_total
            print("dev accuracy: ", dev_acc)

            # update model
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                iters_not_improved = 0
                torch.save(model, "best_model")
            else:
                iters_not_improved += 1
                if iters_not_improved > patience:
                    early_stop = True
                    break


