
local Data = torch.class("Data")

function Data:__init(config)

	-- Alphabet settings
   self.alphabet = config.alphabet or "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
   self.dict = {}
   for i = 1,#self.alphabet do
      self.dict[self.alphabet:sub(i,i)] = i
   end

   self.length = config.length or 100
   self.batch_size = config.batch_size or 128
   self.file = config.file

   self.data = torch.load(self.file)

end

function Data:getBatch(linputs, rinputs, sim_labels, ent_labels)

   local linputs = linputs or torch.Tensor(self.batch_size, self.length)
   local rinputs = rinputs or torch.Tensor(self.batch_size, self.length)
   local sim_labels = sim_labels or torch.Tensor(linputs:size(1))
   local ent_labels = ent_labels or torch.Tensor(linputs:size(1))

   for i = 1, linputs:size(1) do

      -- Choose data
      idx = torch.random(self.data.size)
      sim_score = self.data.sim_labels[idx]
      sim_labels[i] = sim_score
      ent_label = self.data.ent_labels[idx]
      ent_labels[i] = ent_label

      lsent = self.data.lsents[i]
      rsent = self.data.rsents[i]

      linput = self:sent2CharIdx(lsent)
      rinput = self:sent2CharIdx(rsent)

      linputs[i] = linput
      rinputs[i] = rinput

   end

   return linputs, rinputs, sim_labels, ent_labels
end

function Data:iterator(data)
	local i = 1
	local data = data or self.data

	return function()

		if i > #data.lsents then return end

    	local linputs = linputs or torch.Tensor(self.batch_size, #self.alphabet, self.length)
   		local rinputs = rinputs or torch.Tensor(self.batch_size, #self.alphabet, self.length)
   		local sim_labels = sim_labels or torch.Tensor(linputs:size(1))
   		local ent_labels = ent_labels or torch.Tensor(linputs:size(1))

   		local n = 0

   		for k =1, linputs:size(1) do

   			if i > #data.lsents then break end

   			n = n + 1

   			lsent = data.lsents[i]
			rsent = data.rsents[i]
			linputs[k] = self:sent2CharIdx(lsent)
			rinputs[k] = self:sent2CharIdx(rsent)
			sim_labels[k] = self.data.sim_labels[i]
			ent_labels[k] = self.data.ent_labels[i]

			i  = i + 1

		end

    	return linputs, rinputs, sim_labels, ent_labels, n
	end

end


function Data:sent2Tensor(sequence)

  local s = ''
  --print(sequence)
  for i=1, #sequence do
    s = s .. sequence[i] .. " "
  end
  s = s:gsub("%s+", "")
  --trim
  s = s:gsub("^%s*(.-)%s*$", "%1")

  local s = s:lower()
  local t = torch.Tensor(#self.alphabet, self.length)
  t:zero()
  for i = #s, math.max(#s - self.length + 1, 1), -1 do
    if self.dict[s:sub(i,i)] then
      t[self.dict[s:sub(i,i)]][#s - i + 1] = 1
    end
  end
  return t
end

function Data:sent2CharIdx(sequence)

  local s = ''
  --print(sequence)
  for i=1, #sequence do
    s = s .. sequence[i] .. " "
  end
  s = s:gsub("%s+", "")
  --trim
  s = s:gsub("^%s*(.-)%s*$", "%1")

  local s = s:lower()
  local output = torch.Tensor(self.length):fill(#self.alphabet+1)
  for i = #s, math.max(#s - self.length + 1, 1), -1 do
    c = s:sub(i,i)
    if self.dict[c] then
      output[#s-i+1] = self.dict[c]
    else
      output[#s-i+1] = #self.alphabet+1
    end
  end
  return output
end






