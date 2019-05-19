from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

INPUT_CONNECTIONS=5
OUTPUT_CONNECTIONS=5
class Net(nn.Module):
    def __init__(self,initial_neurons,initial_edges):
        super(Net, self).__init__()
        self.neurons=[]
        self.edge_count = initial_edges
        self.neuron_count = initial_neurons
        
            
        
        self.fc2 = nn.Linear(200, 10)
        self.incidence_matrix=np.zeros(shape=(self.edge_count,self.neuron_count,2))
        for row_i in range(len(self.incidence_matrix[:,0,0])):
            target_neuron=np.random.choice(range(1,self.neuron_count),size=1)
            source_neuron=np.random.choice(np.delete(range(0,self.neuron_count-1),target_neuron),size=1)
            self.incidence_matrix[row_i,target_neuron,0]=1
            self.incidence_matrix[row_i,source_neuron,0]=-1

        for neuron_i in range(self.neuron_count):
            input_count = len(np.where(self.incidence_matrix[:,neuron_i,0]==1)[0])
            if input_count ==0:
                input_count=1
            if neuron_i == self.neuron_count-1:
                self.fc1 = nn.Linear(input_count*28*28, 200)
                self.neurons.append(self.fc1)
            else:
                self.neurons.append(nn.Conv2d(input_count, 1, 5, 1,padding=2))
        
        
        self.prepare_for_new_batch()

    def process_neuron(self,neuron_i):
        if neuron_i >0:
            input_edges=np.where(self.incidence_matrix[:,neuron_i,0]==1)[0]
            input_list = np.where(self.incidence_matrix[input_edges,:,0]==-1)[1]
            composed_input=torch.cat([self.current_output_list[input_neuron] for input_neuron in input_list],1)
            if neuron_i == self.neuron_count-1:
                composed_input = composed_input.view(-1, np.prod(composed_input.shape[1:]))
        else:
            composed_input = self.input

        self.future_output_list[neuron_i]=F.relu(self.neurons[neuron_i](composed_input))

        
    def prepare_for_new_batch(self):
        self.future_neuron_process_list = [0]
        self.current_neuron_process_list = []
        self.current_output_list=[torch.zeros((64,1,28,28), requires_grad=True) for  i in range(self.neuron_count)]
        self.future_output_list=self.current_output_list.copy()

    def forward(self, x):
        
        self.processed_signal=np.zeros(self.neuron_count)
        self.input=x
        
        while not self.processed_signal[-1]:
            
            self.current_neuron_process_list=self.future_neuron_process_list.copy()
            self.future_neuron_process_list=[]
            for current_neuron_i in self.current_neuron_process_list:
                self.process_neuron(current_neuron_i)
                self.processed_signal[current_neuron_i]=1
                edges_updated=np.where(self.incidence_matrix[:,current_neuron_i,0]==-1)[0]
                neuron_updated=np.where(self.incidence_matrix[edges_updated,:,0]==1)[1]
                neuron_updated=neuron_updated[self.processed_signal[neuron_updated]==0]
                
                self.future_neuron_process_list.extend(neuron_updated)
                self.future_neuron_process_list=list(set(self.future_neuron_process_list)) #add updated neurons to next neuron process list
            
            self.current_output_list=self.future_output_list.copy()

        #encontrar neurons que já tem todas as entradas
        x = self.current_output_list[-1]
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model.prepare_for_new_batch()
        optimizer.zero_grad()
        for _ in range(10): 
            
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward(retain_graph=True)
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net(15,50).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()