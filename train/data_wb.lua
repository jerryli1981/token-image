--[[
Data Program for Crepe
By Xiang Zhang @ New York University
--]]

local ffi = require("ffi")

-- The class
local Data = torch.class("Data")

function Data:__init(config)
   -- Alphabet settings
   self.alphabet = config.alphabet
   self.dict = {}
   for i = 1,#self.alphabet do
      self.dict[self.alphabet:sub(i,i)] = i
   end


   self.length = config.length
   self.batch_size = config.batch_size or 128
   self.file = config.file
   self.prob = config.prob
   self.thes = config.thes
   self.padding = config.padding
   self.scale = config.scale
   self.extra = config.extra

   self.config = config
   self.data = torch.load(self.file)
   self.num_samples = data.n


end

function Data:nClasses()
   return #self.data.index
end

function Data:getBatch(inputs, labels, data)
   local data = data or self.data
   local inputs = inputs or torch.Tensor(self.batch_size, 4, 5, 5*self.length)

   local labels = labels or torch.Tensor(inputs:size(1))

   for i = 1, inputs:size(1) do
      local label, s
      -- Choose data
      label = torch.random(#data.index)
      local input = torch.random(data.index[label]:size(1))

      s = ffi.string(torch.data(data.content:narrow(1, data.index[label][input][data.index[label][input]:size(1)], 1)))
      for l = data.index[label][input]:size(1) - 1, 1, -1 do
         s = s.." "..ffi.string(torch.data(data.content:narrow(1, data.index[label][input][l], 1)))
      end

      labels[i] = label
      self:sequenceTo3DTensor(s, self.length, inputs:select(1, i))
   end
   return inputs, labels
end

function Data:iterator(static, data)
   local i = 1
   local j = 0
   local data = data or self.data
   local static
   if static == nil then static = true end

   if static then
      inputs = torch.Tensor(self.batch_size, 4, 5, 5*self.length)
      labels = torch.Tensor(inputs:size(1))
   end

   return function()
      if data.index[i] == nil then return end

      local inputs = inputs or torch.Tensor(self.batch_size, 4, 5, 5*self.length)
      local labels = labels or torch.Tensor(inputs:size(1))

      local n = 0
      for k = 1, inputs:size(1) do
         j = j + 1
         if j > data.index[i]:size(1) then
            i = i + 1
            if data.index[i] == nil then
               break
            end
            j = 1
         end

         n = n + 1

         local s = ffi.string(torch.data(data.content:narrow(1, data.index[i][j][data.index[i][j]:size(1)], 1)))
         for l = data.index[i][j]:size(1) - 1, 1, -1 do
            s = s.." "..ffi.string(torch.data(data.content:narrow(1, data.index[i][j][l], 1)))
         end

         self:sequenceTo3DTensor(s, self.length, inputs:select(1, k))
         labels[k] = i
      end
      return inputs, labels, n
   end
end

function Data:sequenceTo3DTensor(str, l, input)

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
         if self.dict[token:sub(i,i)] then 
            word_tensor[i][self.dict[token:sub(i,i)]] = 1
         end
      end
      word_tensor = torch.reshape(word_tensor, 4, 5, 5)

      tmp[count] = word_tensor
      count = count + 1
   end

   if #tmp < 100 then
      for i=#tmp+1, 100 do
         tmp[i] = torch.Tensor(4,5,5):zero()
      end
   end

   local tmp2 = nn.JoinTable(3):forward(tmp)
   for i=1, 4 do
      input[i] = tmp2[i] 
   end
end
