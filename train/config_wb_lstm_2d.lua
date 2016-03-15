--[[
Configuration for Crepe Training Program
By Xiang Zhang @ New York University
--]]

require("nn")
require 'rnn'

-- The namespace
config = {}

local alphabet = "qwertyuiopasdfghjklmxcvbn"

config.seq_length = 400
config.lstmHiddenSize = 64

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
-- 128 x 1600 x 25
config.model[1] = {module = "nn.TemporalConvolution", inputFrameSize = #alphabet, outputFrameSize = 128, kW = 4, dW=4}
config.model[2] = {module = "nn.Threshold"}
-- 128 x 400 x 128
--config.model[3] = {module = "nn.TemporalMaxPooling", kW = 2, dW = 2}
-- 336 x 256
--config.model[4] = {module = "nn.TemporalConvolution", inputFrameSize = 128, outputFrameSize = 128, kW = 4}
--config.model[5] = {module = "nn.Threshold"}
--config.model[6] = {module = "nn.TemporalMaxPooling", kW = 2, dW = 2}

-- 110 x 256
--config.model[7] = {module = "nn.TemporalConvolution", inputFrameSize = 128, outputFrameSize = 128, kW = 4}
--config.model[8] = {module = "nn.Threshold"}
-- 108 x 256
--config.model[9] = {module = "nn.TemporalConvolution", inputFrameSize = 128, outputFrameSize = 128, kW = 4}
--config.model[10] = {module = "nn.Threshold"}
-- 106 x 256
--config.model[11] = {module = "nn.TemporalConvolution", inputFrameSize = 128, outputFrameSize = 128, kW = 4}
--config.model[12] = {module = "nn.Threshold"}
-- 104 x 256
--config.model[13] = {module = "nn.TemporalConvolution", inputFrameSize = 128, outputFrameSize = 128, kW = 4}
--config.model[14] = {module = "nn.Threshold"}
--config.model[15] = {module = "nn.TemporalMaxPooling", kW = 2, dW = 2}


config.model[3] = {module = "nn.SplitTable", dimension = 1, nInputDims = 2}
config.model[4] = {module = "nn.Sequencer", inputSize=128, hiddenSize = config.lstmHiddenSize , seqLength = 400}
--config.model[18] = {module = "nn.JoinTable", dimension = 2}
--config.model[19] = {module = "nn.Reshape", dimension1 = config.train_data.batch_size, dimension2 = 43, dimension3 = config.lstmHiddenSize }
--config.model[20] = {module = "nn.Mean", dimension = 2}
config.model[5] = {module = "nn.SelectTable", dimension = -1}
config.model[6] = {module = "nn.Linear",inputSize = config.lstmHiddenSize, outputSize = 5}
config.model[7] = {module = "nn.LogSoftMax"}

-- The loss
config.loss = nn.ClassNLLCriterion

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


