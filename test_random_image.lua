require 'image'
require 'nn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 30,        -- number of samples to produce
    net = '',              -- path to the generator network
    name = 'test1',        -- name of the experiment and prefix of file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
    nc = 3,                -- # of channels in input
    loadSize = 0,          -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
    fineSize = 64,        -- size of random crops
    nThreads = 1,          -- # of data loading threads to use
    manualSeed = 0,        -- 0 means random seed
    useOverlapPred = 0,        -- overlapping edges (1 means yes, 0 means no). 1 means put 10x more L2 weight on unmasked region.
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

root_dir = '/home/orion/works/Resources/MiddleburyStereoDatasets/scenes2003/FullSize/cones'
output_dir = '/home/orion/Desktop'

print('Load image')
P_path = root_dir .. '/proj2.png'
P = image.load(P_path, nc, 'float')
-- make it [0, 1] -> [-1, 1]
P:mul(2):add(-1)

print('Load mask')
mask_path = root_dir .. '/mask2.png'
M = image.load(mask_path, nc, 'float')
M = torch.lt(M, 0.1):byte()
M = M:sub(1, 1)

local iH = P:size(2)
local iW = P:size(3)
local oH = opt.fineSize
local oW = opt.fineSize
print(P:size())

local n_row, n_col
local ovl_H, ovl_W
local is_use_overlap = true
if is_use_overlap then
   -- Overlap
   ovl_H = 20
   ovl_W = 20
   n_row = math.max(math.floor(iH / ovl_H), math.ceil(iH / ovl_H))
   n_col = math.max(math.floor(iW / ovl_W), math.ceil(iW / ovl_W))
else
   -- Non overlap
   n_row = math.max(math.floor(iH / oH), math.ceil(iH / oH))
   n_col = math.max(math.floor(iW / oW), math.ceil(iW / oW))
end

local batchSize = n_row * n_col
local getBatch = function(P, M)
   Ps = torch.Tensor(batchSize, 3, oH, oW)
   Ms = torch.Tensor(batchSize, 1, oH, oW)
   local index = 1
   for r = 1, n_row do
      for c = 1, n_col do
         -- (x1,y1) is zero-based, (x2, y2) is non-inclusive
         local h1, w1
         if is_use_overlap then
            h1 = math.min((r - 1) * ovl_H, iH - oH)
            w1 = math.min((c - 1) * ovl_W, iW - oW)
         else
            h1 = math.min((r - 1) * oH, iH - oH)
            w1 = math.min((c - 1) * oW, iW - oW)
         end

         --print('c:' .. c .. ' r:' .. r .. ' h1:' .. h1 .. ' w1:' .. w1)
         -- Crop image
         local P_cropped = image.crop(P, w1, h1, w1 + oW, h1 + oH)
         local M_cropped = image.crop(M, w1, h1, w1 + oW, h1 + oH)

         -- Apply mask
         local mask = M_cropped
         P_cropped[{{1},{},{}}][mask] = 2*117.0/255.0 - 1.0
         P_cropped[{{2},{},{}}][mask] = 2*104.0/255.0 - 1.0
         P_cropped[{{3},{},{}}][mask] = 2*123.0/255.0 - 1.0

         -- make it 4D tensor of size [1, 3, fineSize, fineSize]
         Ps[index]:copy(P_cropped)
         Ms[index]:copy(M_cropped)
         index = index + 1
      end
   end
   return Ps, Ms
end

-- load Context-Encoder
assert(opt.net ~= '', 'provide a generator model')
net = util.load(opt.net, opt.gpu)
net:evaluate() -- Set train=false. Dropout, batch_normalization, etc. changes

-- initialize variables
Ps_cuda = torch.Tensor(batchSize, opt.nc, opt.fineSize, opt.fineSize)

-- port to GPU
if opt.gpu > 0 then
    require 'cunn'
    if pcall(require, 'cudnn') then
        print('Using CUDNN !')
        require 'cudnn'
        net = util.cudnn(net)
    end
    net:cuda()
    Ps_cuda = Ps_cuda:cuda()
else
   net:float()
end
print(net)

-- load data
local Ps
local Ms
Ps, Ms = getBatch(P, M) --data:getBatch()
print('Loaded Image Block: ', Ps:size(1)..' x '..Ps:size(2) ..' x '..Ps:size(3)..' x '..Ps:size(4))
Ps_cuda:copy(Ps)

-- run Context-Encoder to inpaint center
local inpainted
inpainted = net:forward(Ps_cuda)

local fillPatch = function(Ps, inpainted, Ms)
   local index = 1
   local filled = Ps:clone()
   for r = 1, n_row do
      for c = 1, n_col do
         local f = filled[{{index},{},{},{}}]
         local m = Ms[{{index},{},{},{}}]:resize(oH, oW):byte()
         local ip = inpainted[{{index},{},{},{}}]
         f[{{},{1},{},{}}][m] = ip[{{},{1},{},{}}][m]:float()
         f[{{},{2},{},{}}][m] = ip[{{},{2},{},{}}][m]:float()
         f[{{},{3},{},{}}][m] = ip[{{},{3},{},{}}][m]:float()
         filled[{{index},{},{},{}}] = f
         index = index + 1
      end
   end
   return filled
end
inpainted = fillPatch(Ps, inpainted, Ms)

-- re-transform scale back to normal
inpainted:add(1):mul(0.5)
local P_merged = torch.Tensor(3, iH, iW)
local M_merged = torch.Tensor(1, iH, iW)
local filled = torch.Tensor(1, iH, iW):fill(1)
filled = torch.gt(filled, 0.1):byte()
local index = 1
for r = 1, n_row do
   for c = 1, n_col do
      local h1, w1
      if is_use_overlap then
         h1 = math.min((r - 1) * ovl_H, iH - oH)
         w1 = math.min((c - 1) * ovl_W, iW - oW)
      else
         h1 = math.min((r - 1) * oH, iH - oH)
         w1 = math.min((c - 1) * oW, iW - oW)
      end
      local fm = filled[{{1}, {h1 + 1, h1 + oH}, {w1 + 1, w1 + oW}}]:byte()
      M_merged[{{1}, {h1 + 1, h1 + oH}, {w1 + 1, w1 + oW}}][fm] = Ms[{{index},{},{},{}}]:resize(1, oH, oW)[fm]

      local fm3 = torch.repeatTensor(fm, 3, 1, 1)
      P_merged[{{1, 3}, {h1 + 1, h1 + oH}, {w1 + 1, w1 + oW}}][fm3] = inpainted[{{index},{},{},{}}]:resize(3, oH, oW)[fm3]

      filled[{{1}, {h1 + 1, h1 + oH}, {w1 + 1, w1 + oW}}] = fm
      index = index + 1
   end
end
image.save(output_dir .. '/tmp_001_projected.png', image.toDisplayTensor(Ps, 0, n_col))
image.save(output_dir .. '/tmp_002_inpainted.png', image.toDisplayTensor(inpainted, 0, n_col))
image.save(output_dir .. '/tmp_003_P_merged.png', P_merged)
image.save(output_dir .. '/tmp_004_M_merged.png', M_merged)
