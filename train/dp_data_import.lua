require 'dp'
require 'torchx' -- for paths.indexdir
--dbg = require("debugger")

local ffi = require("ffi")

alphabet = "qwertyuiopasdfghjklmxcvbn"
dict = {}

for i = 1,#alphabet do
   dict[alphabet:sub(i,i)] = i
end

length = 400

function sequenceTo3DTensor(str, l, input)

   local str = str:lower()
   local count = 1

   local tmp = {}
   for token in string.gmatch(str, "[^%s]+") do

      if count > l then
         break
      end

      local word_tensor = torch.Tensor(4,25)
      word_tensor:zero()

      for i=1, 4 do
         if dict[token:sub(i,i)] then 
            word_tensor[i][dict[token:sub(i,i)]] = 1
         end
      end
      word_tensor = torch.reshape(word_tensor, 4, 5, 5)

      tmp[count] = word_tensor
      count = count + 1
   end

   if #tmp < length then
      for i=#tmp+1, length do
         tmp[i] = torch.Tensor(4,5,5):zero()
      end
   end

   local tmp2 = nn.JoinTable(3):forward(tmp)
   for i=1, 4 do
      input[i] = tmp2[i] 
   end
end

function getInputsAndLabels(data_file)
   data = torch.load(data_file)
   inputs = torch.Tensor(data.size, 4, 5, 5*length)
   labels = torch.IntTensor(data.size):fill(0)

   idx = 1
   for label = 1, #data.index do
      for input=1, data.index[label]:size(1) do
         s = ffi.string(torch.data(data.content:narrow(1, data.index[label][input][data.index[label][input]:size(1)], 1)))
         for l = data.index[label][input]:size(1) - 1, 1, -1 do
            s = s.." "..ffi.string(torch.data(data.content:narrow(1, data.index[label][input][l], 1)))
         end
         labels[idx] = label
         sequenceTo3DTensor(s, length, inputs:select(1, idx))
         idx = idx + 1
      end
   end

   local shuffle = torch.randperm(labels:size(1)) -- shuffle the data

   shuffled_inputs = torch.Tensor(labels:size(1), 4, 5, 5*length)
   shuffled_labels = torch.IntTensor(data.size):fill(0)

   for i=1,labels:size(1) do
      local idx = shuffle[i]
      local img = inputs[idx]
      local lab = labels[idx]
      shuffled_inputs[i]:copy(img)
      shuffled_labels[i]=lab
      collectgarbage()
   end

   return shuffled_inputs, shuffled_labels
end

function getDataset()

   train_data_file = paths.concat(paths.cwd(), "../data/train_wb.t7b")
   test_data_file = paths.concat(paths.cwd(), "../data/test_wb.t7b")

   train_inputs, train_labels = getInputsAndLabels(train_data_file)
   test_inputs, test_labels = getInputsAndLabels(test_data_file)

   train_size = train_inputs:size(1)
   local nValid = math.floor(train_size*0.1)
   local nTrain = train_size - nValid

   local trainInput = dp.ImageView('bchw', train_inputs:narrow(1, 1, nTrain))
   local trainTarget = dp.ClassView('b', train_labels:narrow(1, 1, nTrain))

   local validInput = dp.ImageView('bchw', train_inputs:narrow(1, nTrain+1, nValid))
   local validTarget = dp.ClassView('b', train_labels:narrow(1, nTrain+1, nValid))

   local testInput = dp.ImageView('bchw', test_inputs:narrow(1, 1, test_inputs:size(1)))
   local testTarget = dp.ClassView('b', test_labels:narrow(1, 1, test_inputs:size(1)))

   trainTarget:setClasses({'sports', 'ent', 'auto', 'fin', 'tech'})
   validTarget:setClasses({'sports', 'ent', 'auto', 'fin', 'tech'})
   testTarget:setClasses({'sports', 'ent', 'auto', 'fin', 'tech'})

   local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
   local valid = dp.DataSet{inputs=validInput,targets=validTarget,which_set='valid'}
   local test = dp.DataSet{inputs=testInput,targets=testTarget,which_set='test'}

   -- 4. wrap datasets into datasource
   local ds = dp.DataSource{train_set=train, valid_set=valid, test_set=test}
   ds:classes{'sports', 'ent', 'auto', 'fin', 'tech'}
   return ds
end

getDataset()


