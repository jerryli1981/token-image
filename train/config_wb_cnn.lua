--[[
Configuration for Crepe Training Program
By Xiang Zhang @ New York University
--]]

require("nn")

-- The namespace
config = {}

local alphabet = "qwertyuiopasdfghjklmxcvbn"

seq_length = 100

config.dictsize = #alphabet

-- Training data
config.train_data = {}
config.train_data.file = paths.concat(paths.cwd(), "../data/train_wb.t7b")
config.train_data.alphabet = alphabet
config.train_data.length = seq_length
config.train_data.batch_size = 128

-- Validation data
config.val_data = {}
config.val_data.file =  paths.concat(paths.cwd(), "../data/test_wb.t7b")
config.val_data.alphabet = alphabet
config.val_data.length = seq_length
config.val_data.batch_size = 128

-- The model
config.model = {}
-- 4 x 5 x 500
config.model[1] = {module = "nn.SpatialConvolution", nInputPlane = 4, nOutputPlane= 5, kW = 10, kH=2}
config.model[2] = {module = "nn.Tanh"}
config.model[3] = {module = "nn.SpatialMaxPooling", kW = 3, kH = 2}

-- 34 x 256
config.model[4] = {module = "nn.Reshape", dimension1 = 1630, dimension2 = nil, dimension3 = nil }

config.model[5] = {module = "nn.Linear", inputSize = 1630, outputSize = 5}
config.model[6] = {module = "nn.LogSoftMax"}

-- The loss
config.loss = nn.ClassNLLCriterion

-- The trainer
config.train = {}
local baseRate = 1e-2 * math.sqrt(config.train_data.batch_size) / math.sqrt(128)
config.train.rates = {[1] = baseRate/1,[15001] = baseRate/2,[30001] = baseRate/4,[45001] = baseRate/8,[60001] = baseRate/16,[75001] = baseRate/32,[90001]= baseRate/64,[105001] = baseRate/128,[120001] = baseRate/256,[135001] = baseRate/512,[150001] = baseRate/1024}


config.train.optim = optim.adagrad
config.train.optim_state = {learningRate=config.train.rates[1]}
config.optim_name = "adagrad"

-- The tester
config.test = {}
config.test.confusion = true


-- Main program
config.main = {}
config.main.eras = 1
config.main.epoches = 5000
config.main.randomize = 5e-2
config.main.dropout = true
config.main.save = paths.cwd() .. "/models"
config.main.collectgarbage = 100
config.main.logtime = 5
config.main.validate = true
