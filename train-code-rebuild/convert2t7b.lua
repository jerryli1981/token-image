--[[
Dataset converter from csv to t7b
By Xiang Zhang @ New York University
--]]

require("io")
require("os")
require("paths")
require("torch")

-- Configuration table
config = {}
config.input_dir = "train"
config.output = "train.t7b"

-- Parse arguments
cmd = torch.CmdLine()
cmd:option("-input", config.input_dir, "Input dir")
cmd:option("-output", config.output, "Output t7b file")
params = cmd:parse(arg)
config.input = params.input
config.output = params.output

local dir = 'data/' .. config.input .. '/'

entlabmap = {NEUTRAL=3, CONTRADICTION=1, ENTAILMENT=2}

function read_charsequence(path)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    sentences[#sentences + 1] = tokens
  end

  file:close()
  return sentences
end

local dataset = {}

dataset.lsents = read_charsequence(dir .. 'a.toks')
dataset.rsents = read_charsequence(dir .. 'b.toks')

dataset.size = #dataset.lsents

local sim_file = torch.DiskFile(dir .. 'sim.txt', 'r')
local ent_file = io.open(dir .. 'ent.txt', 'r')
dataset.sim_labels = torch.Tensor(dataset.size)
dataset.ent_labels = torch.Tensor(dataset.size)


for i = 1, dataset.size do
 dataset.sim_labels[i] = 0.25 * (sim_file:readDouble() - 1)
 dataset.ent_labels[i] = entlabmap[ent_file:read()]
end

sim_file:close()
ent_file:close()


print("Saving to "..'data/' .. config.output)
torch.save('data/' .. config.output, dataset)
print("Processing done")