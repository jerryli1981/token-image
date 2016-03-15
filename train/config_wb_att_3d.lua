--[[
Configuration for Crepe Training Program
By Xiang Zhang @ New York University
--]]

require("nn")

-- The namespace
config = {}

local alphabet = "qwertyuiopasdfghjklmxcvbn"

config.seq_length = 400

config.dictsize = #alphabet

-- Training data
config.train_data = {}
config.train_data.file = paths.concat(paths.cwd(), "../data/train_wb.t7b")
config.train_data.alphabet = alphabet
config.train_data.length = config.seq_length
config.train_data.batch_size = 128

-- Validation data
config.val_data = {}
config.val_data.file =  paths.concat(paths.cwd(), "../data/test_wb.t7b")
config.val_data.alphabet = alphabet
config.val_data.length = config.seq_length
config.val_data.batch_size = 128

-- The model
config.model = {}
-- The model
config.model = {}
-- 4 x 5 x (5 x seq_length)
config.model[1] = {module = "nn.SpatialConvolution", nInputPlane = 4, nOutputPlane= 256, kW = 20, kH=5, dW = 5, dH = 1}
config.model[2] = {module = "nn.ReLU"}
config.model[3] = {module = "nn.SpatialMaxPooling", kW = 2, kH = 1, dW=2, dH=1}

config.model[4] = {module = "nn.SpatialConvolution", nInputPlane = 256, nOutputPlane= 256, kW = 20, kH=1, dW = 5, dH = 1}
config.model[5] = {module = "nn.ReLU"}
config.model[6] = {module = "nn.SpatialMaxPooling", kW = 2, kH = 1, dW=2, dH=1}


config.model[7] = {module = "nn.Reshape", dimension1 = 2048, dimension2 = nil, dimension3 = nil }

config.model[8] = {module = "nn.Linear", inputSize = 2048, outputSize = 1024}
config.model[9] = {module = "nn.ReLU"}
config.model[10] = {module = "nn.Dropout", p = 0.5}

config.model[11] = {module = "nn.Linear", inputSize = 1024, outputSize = 5}
config.model[12] = {module = "nn.LogSoftMax"}


-- The loss
config.loss = nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
      :add(nn.ModuleCriterion(nn.VRClassReward(agent, opt.rewardScale), nil, nn.Convert()))

-- The trainer
config.train = {}
local baseRate = 1e-2 * math.sqrt(config.train_data.batch_size) / math.sqrt(128)
config.train.rates = {[1] = baseRate/1,[15001] = baseRate/2,[30001] = baseRate/4,[45001] = baseRate/8,[60001] = baseRate/16,[75001] = baseRate/32,[90001]= baseRate/64,[105001] = baseRate/128,[120001] = baseRate/256,[135001] = baseRate/512,[150001] = baseRate/1024}
config.train.momentum = 0.9
config.train.decay=1e-5

--config.train.optim = optim.adagrad
--config.train.optim_state = {learningRate=config.train.rates[1]}
--config.optim_name = "adagrad"

--config.train.optim = optim.sgd
--config.train.optim_state = {momentum = 0.9, weightDecay = 1e-5, learningRate=config.train.rates[1]}
config.optim_name = "sgd"

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
