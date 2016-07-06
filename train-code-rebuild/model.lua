--[[
Model Program for Crepe
By Xiang Zhang @ New York University
--]]

-- Prerequisite
require("nn")
require("nngraph")

-- The class
local Model = torch.class("Model")

function Model:__init(config, tensortype)
   -- Create a sequential for self
   self.graph = Model:createGraph(config)
   self.p = config.p or 0.5
   self.tensortype = torch.getdefaulttensortype()
end

-- Get the parameters of the model
function Model:getParameters()
   return self.graph:getParameters()

end

-- Forward propagation
function Model:forward(linput, rinput)
   self.output = self.graph:forward({linput, rinput})
   return self.output
end

-- Backward propagation
function Model:backward(linput, rinput, gradOutput)
   self.gradInput = self.graph:backward({linput, rinput}, gradOutput)
   return self.gradInput
end


-- Randomize the model to random parameters
function Model:randomize(sigma)
   local w,dw = self:getParameters()
   w:normal():mul(sigma or 1)
end

-- Enable Dropouts
function Model:enableDropouts()
   self.graph = self:changeDropouts(self.graph, self.p)
end

-- Disable Dropouts
function Model:disableDropouts()
   self.graph = self:changeDropouts(self.graph,0)
end

-- Switch to a different data mode
function Model:type(tensortype)
   if tensortype ~= nil then
      self.graph = self:makeCleanGraph(self.graph)
      self.graph:type(tensortype)
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
function Model:changeDropouts(model,p)

   seqModel_1 = model:findModules("nn.Sequential")[1]
   for i,m in ipairs(seqModel_1.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
         m.p = p
      end
   end

   seqModel_2 = model:findModules("nn.Sequential")[2]
   for i,m in ipairs(seqModel_2.modules) do
      if m.module_name == "nn.Dropout" or torch.typename(m) == "nn.Dropout" then
         m.p = p
      end
   end

   return model
end

-- Create a sequential model using configurations
function Model:createGraph(model)

   local linput, rinput = nn.Identity()(), nn.Identity()()

   local left = nn.Sequential()
   local right = nn.Sequential()
   for i, m in ipairs(model) do
      if i < 25 then
         left:add(Model:createModule(m))
         right:add(Model:createModule(m))
      end
   end

   lvec = left(linput)

   rvec = right(rinput)

   --local mult_dist = nn.CMulTable(){lvec, rvec}
   local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
   --local vec_dist_feats = nn.JoinTable(2){lvec, rvec}
   local vecs_to_input = nn.gModule({linput, rinput}, {add_dist})

   -- define similarity model architecture
   local sim_module = nn.Sequential()
      :add(vecs_to_input)

   for i, m in ipairs(model) do
      if i >= 25 then
         sim_module:add(Model:createModule(m))
      end
   end

   return sim_module

end

function Model:makeCleanGraph(model)

   local linput, rinput = nn.Identity()(), nn.Identity()()

   local left = nn.Sequential()
   local right = nn.Sequential()

   gmod = model:findModules("nn.gModule")[1]   
   seqModel = gmod:findModules("nn.Sequential")[1]

   for i = 1, #seqModel.modules do
      local m = Model:makeCleanModule(seqModel.modules[i])
      if m then
         left:add(m)
         right:add(m)
      end
   end

   lvec = left(linput)
   rvec = right(rinput)

   --local mult_dist = nn.CMulTable(){lvec, rvec}
   --local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
   local vec_dist_feats = nn.JoinTable(2){lvec, rvec}
   local vecs_to_input = nn.gModule({linput, rinput}, {vec_dist_feats})

   -- define similarity model architecture

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

-- Create a module using configurations
function Model:createModule(m)
   if m.module == "nn.Reshape" then
      return Model:createReshape(m)
   elseif m.module == "nn.Linear" then
      return Model:createLinear(m)
   elseif m.module == "nn.Threshold" then
      return Model:createThreshold(m)
   elseif m.module == "nn.Sigmoid" then
      return Model:createSigmoid(m)
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
   else
      error("Unrecognized module for creation: "..tostring(m.module))
   end
end

-- Make a clean module
function Model:makeCleanModule(m)
   if torch.typename(m) == "nn.TemporalConvolution" then
    return Model:toTemporalConvolution(m)
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
   elseif torch.typename(m) == "nn.Sigmoid" then
      return Model:newSigmoid()
   elseif torch.typename(m) == "nn.LookupTable" then
      return Model:toLookupTable(m)
   else
      error("Module unrecognized")
   end
end

-- Create a new reshape model
function Model:createReshape(m)
   return nn.Reshape(m.size)
end

-- Create a new linear model
function Model:createLinear(m)
   return nn.Linear(m.inputSize, m.outputSize)
end

-- Create a new threshold model
function Model:createThreshold(m)
   return nn.Threshold()
end

-- Create a new threshold model
function Model:createSigmoid(m)
   return nn.Sigmoid()
end

-- Create a new Spatial Convolution model
function Model:createTemporalConvolution(m)
   return nn.TemporalConvolution(m.inputFrameSize, m.outputFrameSize, m.kW, m.dW)
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

-- Create new LookupTable module
function Model:createLookupTable(m)
   return nn.LookupTable(m.char_vocab_size, m.inputFrameSize)
end

-- Create a new threshold
function Model:newThreshold()
   return nn.Threshold()
end

-- Create a new Sigmoid
function Model:newSigmoid()
   return nn.Sigmoid()
end

-- Convert to a new max pooling
function Model:toTemporalMaxPooling(m)
   return nn.TemporalMaxPooling(m.kW, m.dW)
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


-- Create a new LogSoftMax
function Model:toLookupTable(m)
   local new = nn.LookupTable(m.weight:size(1), m.weight:size(2))
   new.weight:copy(m.weight)
   return new
end

-- Convert a convolution module to standard
function Model:toTemporalConvolution(m)
   local new = nn.TemporalConvolution(m.inputFrameSize, m.outputFrameSize, m.kW, m.dW)
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end