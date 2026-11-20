# 机制诊断与开发粗筛脚本

状态：代码已编写，轻量单元测试通过；尚未加载真实模型运行诊断、拟合新连接或启动开发搜索。真实GPU集成、性能、最终指标尚待执行验证。

FC新旧Hebbian与旧卷积短时Hebbian关闭，网络权重冻结。训练仅原fold1–3，开发从fold4每类2概念×2图，fold0不参与。所有视觉角色的概念/字节隔离有断言。历史数据开发不称独立确认。

78候选：rep4加性28；prd5加性28；rep4门控12；rep4乘性4；额外门控无Hebbian6。prd5保留误差梯度链；rep4在最终表征处求预测误差梯度。共同PCoder4截断基线。乘性分母tau=1（训练RMS单位），不按测试选择。柱间使用掩码稠密卷积参考，尚无稀疏加速核。

主脚本默认plan只显示候选和数据数量，不提取、不写结果。

```powershell
$py='C:/Users/liang/anaconda3/envs/predify2021/python.exe'
& $py analysis/conv_development_20261104.py plan --dataset imagenet
& $py analysis/conv_development_20261104.py diagnose --dataset imagenet
& $py analysis/conv_development_20261104.py screen --dataset imagenet --candidate c000
& $py analysis/conv_development_20261104.py screen --dataset imagenet --candidate c027
& $py analysis/evaluate_conv_development_20261104.py --dataset imagenet --candidate c027
```

THINGS使用--dataset things。c000为仅截断基线；在相同site、lambda条件下选gain=0候选作为仅抑制。候选完整参数以plan输出为准。暂不提供自动择优或无上限搜索：先检查诊断和基线，再显式提交候选。

诊断保存rep4/5、prd4/5每步的mean/min/精确零值比例/相对当前张量RMS的1e-3活动比例/预测误差/0.1%、1%、50%、99%分位数；该活动比例仅用于分布诊断，不是训练固定阈值的最终稀疏性指标。保存preview和target诊断，主CHW/FC特征保存target15步。粗筛只用exact/different、clean/固定noisy。

评价脚本从内层训练概念的clean reference t9重拟合并冻结基线分类头，不复用包含开发样本的旧fold0头。输出四层完整CHW几何、精确零值比例、平均激活、FC2参考ACC；early/late的Exact−Different差值。比较不同候选的相同seed/layer/window获得相对clip/抑制差中差。适配读出、置信区间、Same-concept/category、自动多目标排名和跨折复核属于后续阶段。

结果位于Hebbian-CNN/Log/conv_development_20261104。保存源代码hash与split；源或split改变后拒绝继续混写。单条结果原子写入，可续跑。

轻量测试：analysis/test_conv_development_20261104.py。当前未启动后台任务。
