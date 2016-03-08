--[[
Model Program for Crepe
By Xiang Zhang @ New York University
--]]

-- Prerequisite
require("nn")
require("nngraph")

-- The class
local Model = torch.class("Model")

function Model:__init(config)

   -- Create a sequential for self
   if config.file then
      --self.sequential = Model:makeCleanSequential(torch.load(config.file))
      self.sequential = Model:makeCleanParallel(torch.load(config.file))
   else
      --self.sequential = Model:createSequential(config)
      self.sequential = Model:createParallel(config)
   end
   self.p = config.p or 0.5
   self.tensortype = torch.getdefaulttensortype()
   
end

-- Get the parameters of the model
function Model:getParameters()
   return self.sequential:getParameters()
end

-- Forward propagation
function Model:forward(input)
   self.output = self.sequential:forward(input)
   return self.output
end

-- Backward propagation
function Model:backward(input, gradOutput)
   self.gradInput = self.sequential:backward(input, gradOutput)
   return self.gradInput
end

-- Randomize the model to random parameters
function Model:randomize(sigma)
   local w,dw = self:getParameters()
   w:normal():mul(sigma or 1)
end

-- Enable Dropouts
function Model:enableDropouts()
   self.sequential = self:changeSequentialDropouts(self.sequential, self.p)
end

-- Disable Dropouts
function Model:disableDropouts()
   self.sequential = self:changeSequentialDropouts(self.sequential,0)
end

-- Switch to a different data mode
function Model:type(tensortype)
   if tensortype ~= nil then
      --self.sequential = self:makeCleanSequential(self.sequential)
      self.sequential = self:makeCleanParallel(self.sequential)
      self.sequential:type(tensortype)
      self.tensortype = tensortype
   end
   return self.tensortype
end

-- Switch to cuda
function Model:cuda()
   self:type("torch.CudaTensor")
end

-- Switch to double
function Model:double()
   self:type("torch.DoubleTensor")
end

-- Switch to float
function Model:float()
   self:type("torch.FloatTensor")
end

-- Change dropouts
function Model:changeSequentialDropouts(model,p)
   for i,m in ipairs(model.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
	   m.p = p
      end
   end
   return model
end

-- Create a parallel model using configurations
function Model:createParallel(model)

   local input = nn.Identity()()

   window_size =4

   inputs = {}
   for i=1, 5*config.seq_length, 5*window_size do
      ipt = nn.Narrow(4, i, 5*window_size)(input)  
      table.insert(inputs, ipt)
   end
   sequential_list = {}
   for i=1, config.seq_length/window_size do
      seq = nn.Sequential()
      table.insert(sequential_list, seq)
   end

   for i, m in ipairs(model) do
      if i < 7 then
         for j=1, config.seq_length/window_size do
            seq = sequential_list[j]
            seq:add(Model:createModule(m))
         end
      end
   end

   cnn=nn.ParallelTable()
   for j=1, config.seq_length/window_size do
      seq = sequential_list[j]
      cnn:add(seq)
   end

   cnn_out = cnn(inputs)

   merged = nn.JoinTable(4){cnn_out}
   
   local vecs_to_input = nn.gModule({input}, {merged})

   -- define similarity model architecture

   local sim_module = nn.Sequential()
      :add(vecs_to_input)

   for i, m in ipairs(model) do
      if i >= 7 then
         sim_module:add(Model:createModule(m))
      end
   end

   return sim_module

end

function Model:makeCleanParallel(model)

   local input = nn.Identity()()
   window_size =4

   inputs = {}
   for i=1, 5*config.seq_length, 5*window_size do
      ipt = nn.Narrow(4, i, 5*window_size)(input)
      table.insert(inputs, ipt)
   end

   sequential_list = {}
   for i=1, config.seq_length/window_size do
      seq = nn.Sequential()
      table.insert(sequential_list, seq)
   end

   gmod = model:findModules("nn.gModule")[1]   
   seqModel = gmod:findModules("nn.Sequential")[1]


   for i = 1, #seqModel.modules do
      local m = Model:makeCleanModule(seqModel.modules[i])
      if m then
         for j=1, config.seq_length/window_size do
            seq = sequential_list[j]
            seq:add(m)
         end
      end
   end

   cnn=nn.ParallelTable()
   for j=1, config.seq_length/window_size do
      seq = sequential_list[j]
      cnn:add(seq)
   end

   cnn_out = cnn(inputs)
   merged = nn.JoinTable(4){cnn_out}

   local vecs_to_input = nn.gModule({input}, {merged})
   
   sim_m = model:findModules("nn.Sequential")[1]
   local sim_module = nn.Sequential()
   sim_module:add(vecs_to_input)
   for i=2, #sim_m.modules do
      local m = Model:makeCleanModule(sim_m.modules[i])
      if m then
         sim_module:add(m)
      end
   end
   return sim_module 

end

-- Create a sequential model using configurations
function Model:createSequential(model)
   local new = nn.Sequential()
   for i,m in ipairs(model) do
      new:add(Model:createModule(m))
   end
   return new
end

-- Clear the module out of gradient data and input/output
function Model:clearSequential(model)
   for i,m in ipairs(model.modules) do
      if m.output then m.output = torch.Tensor() end
      if m.gradInput then m.gradInput = torch.Tensor() end
      if m.gradWeight then m.gradWeight = torch.Tensor() end
      if m.gradBias then m.gradBias = torch.Tensor() end
   end
   return model
end

-- Make a clean sequential model
function Model:makeCleanSequential(model)
   local new = nn.Sequential()
   for i = 1,#model.modules do
      local m = Model:makeCleanModule(model.modules[i])
      if m then
	 new:add(m)
      end
   end
   return new
end

-- Create a module using configurations
function Model:createModule(m)
   if m.module == "nn.Reshape" then
      return Model:createReshape(m)
   elseif m.module == "nn.Linear" then
      return Model:createLinear(m)
   elseif m.module == "nn.Threshold" then
      return Model:createThreshold(m)
   elseif m.module == "nn.TemporalConvolution" then
      return Model:createTemporalConvolution(m)
   elseif m.module == "nn.TemporalMaxPooling" then
      return Model:createTemporalMaxPooling(m)
   elseif m.module == "nn.Dropout" then
      return Model:createDropout(m)
   elseif m.module == "nn.LogSoftMax" then
      return Model:createLogSoftMax(m)
   elseif m.module == "nn.LookupTable" then
      return Model:createLookupTable(m)
   elseif m.module == "nn.SplitTable" then
      return Model:createSplitTable(m)
   elseif m.module == "nn.JoinTable" then
      return Model:createJoinTable(m)
   elseif m.module == "nn.Transpose" then
      return Model:createTranspose(m)
   elseif m.module == "nn.Tanh" then
      return Model:createTanh(m)
   elseif m.module == "nn.ReLU" then
      return Model:createReLU(m)
   elseif m.module == "nn.Sequencer" then
      return Model:createSequencer(m)
   elseif m.module == "nn.Mean" then
      return Model:createMean(m)
   elseif m.module == "nn.SpatialConvolution" then
      return Model:createSpatialConvolution(m)
   elseif m.module == "nn.SpatialMaxPooling" then
      return Model:createSpatialMaxPooling(m)
   else
      error("Unrecognized module for creation: "..tostring(m.module))
   end
end

-- Make a clean module
function Model:makeCleanModule(m)
   if torch.typename(m) == "nn.TemporalConvolution" then
	 return Model:toTemporalConvolution(m)
   elseif torch.typename(m) == "nn.SpatialConvolution" then
      return Model:toSpatialConvolution(m)
   elseif torch.typename(m) == "nn.SpatialMaxPooling" then
      return Model:toSpatialMaxPooling(m)
   elseif torch.typename(m) == "nn.Threshold" then
      return Model:newThreshold()
   elseif torch.typename(m) == "nn.TemporalMaxPooling" then
      return Model:toTemporalMaxPooling(m)
   elseif torch.typename(m) == "nn.Reshape" then
      return Model:toReshape(m)
   elseif torch.typename(m) == "nn.Linear" then
      return Model:toLinear(m)
   elseif torch.typename(m) == "nn.LogSoftMax" then
      return Model:newLogSoftMax(m)
   elseif torch.typename(m) == "nn.Dropout" then
      return Model:toDropout(m)
   elseif torch.typename(m) == "nn.LookupTable" then
      return Model:toLookupTable(m)
   elseif torch.typename(m) == "nn.Transpose" then
      return Model:toTranspose(m)
   elseif torch.typename(m) == "nn.SplitTable" then
      return Model:toSplitTable(m)
   elseif torch.typename(m) == "nn.JoinTable" then
      return Model:toJoinTable(m)
   elseif torch.typename(m) == "nn.Sequencer" then
      return Model:toSequencer(m)
   elseif torch.typename(m) == "nn.Mean" then
      return Model:toMean(m)
   elseif torch.typename(m) == "nn.Tanh" then
      return Model:newTanh()
   elseif torch.typename(m) == "nn.ReLU" then
      return Model:newReLU()
   else
      error("Module unrecognized")
   end
end

function Model:createSequencer(m)
   --local r = nn.Recurrent(m.hiddenSize, nn.Identity(), 
      --nn.Linear(m.hiddenSize, m.hiddenSize), nn.Sigmoid(), m.seqLength)
   local r = nn.FastLSTM(m.inputSize, m.hiddenSize)
   return nn.Sequencer(r)
end


function Model:createTranspose(m)
   return nn.Transpose(m.dimension_1, m.dimension_2)
end

function Model:createMean(m)
   return nn.Mean(m.dimension)
end

function Model:createSplitTable(m)
   return nn.SplitTable(m.dimension, m.nInputDims)
end

function Model:createJoinTable(m)
   return nn.JoinTable(m.dimension)
end


-- Create new LookupTable module
function Model:createLookupTable(m)
   return nn.LookupTable(m.char_vocab_size, m.inputFrameSize)
end

function Model:createTanh(m)
   return nn.Tanh()
end

function Model:createReLU(m)
   return nn.ReLU()
end


-- Create a new reshape model
function Model:createReshape(m)
   return nn.Reshape(m.dimension1, m.dimension2, m.dimension3)
end

-- Create a new linear model
function Model:createLinear(m)
   return nn.Linear(m.inputSize, m.outputSize)
end

-- Create a new threshold model
function Model:createThreshold(m)
   return nn.Threshold()
end

-- Create a new Spatial Convolution model
function Model:createTemporalConvolution(m)
   return nn.TemporalConvolution(m.inputFrameSize, m.outputFrameSize, m.kW, m.dW)
end

function Model:createSpatialConvolution(m)
   return nn.SpatialConvolution(m.nInputPlane, m.nOutputPlane, m.kW, m.kH, m.dW, m.dH)
end

function Model:createSpatialMaxPooling(m)
   return nn.SpatialMaxPooling(m.kW, m.kH, m.dW, m.dH)
end

-- Create a new spatial max pooling model
function Model:createTemporalMaxPooling(m)
   return nn.TemporalMaxPooling(m.kW, m.dW, m.dH)
end

-- Create a new dropout module
function Model:createDropout(m)
   return nn.Dropout(m.p)
end

-- Create new logsoftmax module
function Model:createLogSoftMax(m)
   return nn.LogSoftMax()
end

-- Create a new threshold
function Model:newThreshold()
   return nn.Threshold()
end

function Model:newTanh()
   return nn.Tanh()
end

function Model:newReLU()
   return nn.ReLU()
end

-- Convert to a new max pooling
function Model:toTemporalMaxPooling(m)
   return nn.TemporalMaxPooling(m.kW, m.dW)
end

function Model:toSpatialMaxPooling(m)
   return nn.SpatialMaxPooling(m.kW, m.kH, m.dW, m.dH)
end

-- Convert to a new reshape
function Model:toReshape(m)
   return nn.Reshape(m.size)
end

-- Convert to a new dropout
function Model:toDropout(m)
   return nn.Dropout(m.p)
end

-- Convert to a new linear module
function Model:toLinear(m)
   local new = nn.Linear(m.weight:size(2),m.weight:size(1))
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end

-- Create a new LogSoftMax
function Model:newLogSoftMax()
   return nn.LogSoftMax()
end

-- Convert a convolution module to standard
function Model:toTemporalConvolution(m)
   local new = nn.TemporalConvolution(m.inputFrameSize, m.outputFrameSize, m.kW, m.dW)
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end

function Model:toSpatialConvolution(m)
   local new = nn.SpatialConvolution(m.nInputPlane, m.nOutputPlane, m.kW, m.kH, m.dW, m.dH)
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end

-- Create a new LogSoftMax
function Model:toLookupTable(m)
   local new = nn.LookupTable(m.weight:size(1), m.weight:size(2))
   new.weight:copy(m.weight)
   return new
end

function Model:toTranspose(m)
   local new = nn.Transpose(m.dimension_1, m.dimension_2)
   return new
end

function Model:toSplitTable(m)
   local new = nn.SplitTable(m.dimension, m.nInputDims)
   return new
end

function Model:toJoinTable(m)
   local new = nn.JoinTable(m.dimension)
   return new
end

function Model:toMean(m)
   return nn.Mean(m.dimension)
end

function Model:toSequencer(m)
   local r = m.module
   return nn.Sequencer(r)

end
