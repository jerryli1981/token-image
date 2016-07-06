require("xlua")

local Train = torch.class("Train")


function Train:__init(data, model, loss, config)
	self.data =data
	self.model = model
	self.loss = loss

	self.rates = config.rates or {1e-3}
	self.epoch = 1

	self.params, self.grads = self.model:getParameters()
	self.old_grads = self.grads:clone():zero()

	self.loss:type(self.model:type())

	-- Find the current rate
	local max_epoch = 1
	self.rate = self.rates[1]
	for i,v in pairs(self.rates) do
	  if i <= self.epoch and i > max_epoch then
	 max_epoch = i
	 self.rate = v
	  end
	end

	-- Store the configurations
	self.momentum = config.momentum or 0
	self.decay = config.decay or 0
	self.normalize = config.normalize
	self.recapture = config.recapture
end

function Train:run(epochs)
	for i = 1, epochs do
		xlua.progress(i, epochs)
		self:batchStep()
	end
end

function Train:batchStep()

	linputs, rinputs, _, ent_labels = self.data:getBatch()
	
	--self.lbatch = self.lbatch or linputs:transpose(2,3):contiguous():type(self.model:type())
	--self.rbatch = self.rbatch or rinputs:transpose(2,3):contiguous():type(self.model:type())

	self.lbatch = self.lbatch or linputs:type(self.model:type())
	self.rbatch = self.rbatch or rinputs:type(self.model:type())

	self.ent_labels = self.ent_labels or ent_labels:type(self.model:type())

	--self.lbatch:copy(linputs:transpose(2,3):contiguous())
	--self.rbatch:copy(rinputs:transpose(2,3):contiguous())

	self.lbatch:copy(linputs)
	self.rbatch:copy(rinputs)

	self.ent_labels:copy(ent_labels)

	self.output = self.model:forward(self.lbatch, self.rbatch)
	self.objective = self.loss:forward(self.output, self.ent_labels)

	-- Backward propagation   
	self.grads:zero()
	self.gradOutput = self.loss:backward(self.output,self.ent_labels)
	self.gradBatch = self.model:backward(self.lbatch,self.rbatch,self.gradOutput)


	if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end

	-- Update the step
	self.old_grads:mul(self.momentum):add(self.grads:mul(-self.rate))
	self.params:mul(1-self.rate*self.decay):add(self.old_grads)
	if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end

	-- Increment on the epoch
	self.epoch = self.epoch + 1
	-- Change the learning rate
	self.rate = self.rates[self.epoch] or self.rate

end