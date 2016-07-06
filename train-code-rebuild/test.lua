local Test = torch.class("Test")

function Test:__init(data, model, loss, config)
	local config = config or {}

	self.data = data
	self.model = model
	self.loss =loss

	self.loss:type(model:type())

end

function Test:run()
	self.e = 0
	self.n = 0

	for lbatch, rbatch, _, ent_labels, n in self.data:iterator() do

		--self.lbatch = self.lbatch or lbatch:transpose(2,3):contiguous():type(self.model:type())
		--self.rbatch = self.rbatch or rbatch:transpose(2,3):contiguous():type(self.model:type())

		self.lbatch = self.lbatch or linputs:type(self.model:type())
		self.rbatch = self.rbatch or rinputs:type(self.model:type())

		self.ent_labels = self.ent_labels or ent_labels:type(self.model:type())

		--self.lbatch:copy(lbatch:transpose(2,3):contiguous())
		--self.rbatch:copy(rbatch:transpose(2,3):contiguous())

		self.lbatch:copy(linputs)
		self.rbatch:copy(rinputs)

		self.ent_labels:copy(ent_labels)

		self.output = self.model:forward(self.lbatch, self.rbatch)

		self.max, self.decision = self.output:double():max(2)
	    self.max = self.max:squeeze():double()
	    self.decision = self.decision:squeeze():double()
	    self.err = torch.ne(self.decision,self.ent_labels:double()):sum()/self.ent_labels:size(1)

	    self.e = self.e*(self.n/(self.n+n)) +  self.err*(n/(self.n+n))

	    self.n = self.n + n

	    if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
	end
	
	return self.e

end

