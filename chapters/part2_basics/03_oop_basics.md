# 第三章：Python 面向对象基础 — 为理解 nn.Module 铺路

## 3.0 本章导引

### 这一章不讲 PyTorch

这一章讲的是**纯 Python**。你可能会觉得奇怪——明明是 PyTorch 教程，为什么花一整章讲 Python 类？

因为：

```
PyTorch 中所有神经网络都是 Python 类。

class MyNetwork(nn.Module):    ← 这是类
    def __init__(self):        ← 这是方法
        super().__init__()     ← 这需要理解继承
        self.fc = nn.Linear()  ← 这需要理解 self

如果你不理解 class、self、__init__、super()、继承：
    → 你永远在"模仿代码"而不是"自己写代码"
    → 每个网络的 __init__ 和 forward 对你来说都是"固定写法"
    → 你无法理解为什么 model(x) 能工作

如果你理解了它们：
    → nn.Module 不再神秘
    → 你能自信地设计任意结构的网络
    → PyTorch 的"魔法"变成了"合理的设计"
```

### 本教程区别于所有其他 PyTorch 教程的地方

大多数 PyTorch 教程默认你**已经**懂面向对象编程。但实际上很多初学者根本不懂。

本章就是为这些初学者准备的。

### 本章地图

```
3.1 为什么需要类            ← 从散装变量 → 类的心路历程
3.2 class 是什么            ← 蓝图与房子
3.3 __init__ 和 self        ← 最核心的两个概念
3.4 方法                    ← 类里面的函数
3.5 完整示例：计数器          ← 函数版 vs 类版
3.6 继承                    ← 为什么 nn.Module 是父类
3.7 super()                 ← 为什么必须 super().__init__()
3.8 方法重写                ← forward() 就是重写
3.9 对照 PyTorch            ← 把 OOP 概念映射到 nn.Module
3.10 练习
```

> 如果你已经非常熟悉 Python 类，**至少读 3.6-3.9 节**——它们专门为理解 `nn.Module` 做了映射。

---

## 3.1 为什么需要类

### 3.1.1 从一段"看起来还行"的代码开始

假设你在写一个简单的游戏，有两个角色：

```python
# %%
# 散装变量管理两个角色
player1_name = "战士"
player1_hp = 100
player1_attack = 20

player2_name = "法师"
player2_hp = 70
player2_attack = 35

# 攻击函数
def attack(attacker_name, attacker_atk, defender_name, defender_hp):
    new_hp = defender_hp - attacker_atk
    print(f"{attacker_name} 攻击了 {defender_name}！")
    print(f"{defender_name} 剩余 HP: {new_hp}")
    return new_hp

# 战士攻击法师
player2_hp = attack(player1_name, player1_attack, 
                    player2_name, player2_hp)
```

**这段代码有什么问题？**

1. **数据和操作分离**：角色数据（`player1_name`、`player1_hp`...）和攻击逻辑（`attack` 函数）是分开定义的。当你添加第三个角色时，需要再加 3 个变量 + 更长的函数调用。

2. **容易传错参数**：`attack(player1_name, player2_attack, ...)` — 如果不小心把不同角色的 name 和 attack 混在一起，Python 不会报错，但逻辑就错了。

3. **无法保证数据一致性**：任何人都可以写 `player1_hp = -999`，没有机制阻止。

### 3.1.2 用字典能好一点，但不够

```python
# %%
player1 = {"name": "战士", "hp": 100, "attack": 20}
player2 = {"name": "法师", "hp": 70, "attack": 35}

def attack(attacker, defender):
    defender["hp"] -= attacker["attack"]
    print(f"{attacker['name']} → {defender['name']}")

attack(player1, player2)
```

**稍有改善**，但仍有致命缺陷：

- 别人可能写 `player1["HP"]`（大写），就出 bug 了——字典没有"结构约束"
- 攻击函数和角色数据**仍然是分开的**——你在代码的任何角落都能改 `player1["hp"]`

### 3.1.3 类的解决方案

```python
# %%
class Character:
    def __init__(self, name, hp, attack):
        self.name = name
        self.hp = hp
        self.attack = attack
    
    def attack_enemy(self, enemy):
        enemy.hp -= self.attack
        print(f"{self.name} 攻击了 {enemy.name}！")
        print(f"{enemy.name} 剩余 HP: {enemy.hp}")

# 使用
hero = Character("战士", 100, 20)
monster = Character("哥布林", 50, 10)

hero.attack_enemy(monster)   # 战士攻击哥布林
```

**对比**：

```
散装变量版：
    数据：player1_name, player1_hp, player1_attack (3 个独立变量)
    操作：attack(attacker_name, attacker_atk, defender_name, defender_hp)
    问题：数据和行为分离，传参容易出错

类版本：
    数据+行为：Character 类把 name/hp/attack 和 attack_enemy 打包在一起
    hero.attack_enemy(monster)  ← 调用自然，不会传错参数
```

### 3.1.4 为什么神经网络天然适合用类

一个神经网络：

```
有状态（参数）：  W₁, b₁, W₂, b₂, W₃, b₃ ...
有行为：        forward(x) → 接收输入，计算输出

这不就是"数据 + 行为"的完美组合吗？
```

```
类比：
    Character 类：    状态 = (name, hp, attack)，行为 = attack_enemy()
    神经网络类：       状态 = (W₁, b₁, W₂, b₂...)，行为 = forward(x)
```

---

## 3.2 class —— 定义一个类

### 3.2.1 最简类——3 行代码

```python
# %%
class Character:
    pass

# 创建两个具体角色（对象 / 实例）
hero = Character()
monster = Character()

print(type(hero))           # <class '__main__.Character'>
print(type(monster))        # <class '__main__.Character'>
print(hero is monster)      # False — 两个独立的对象
```

**逐行解释：**

| 代码 | 解释 |
|------|------|
| `class Character:` | 定义了一个名为 `Character` 的类。`class` 是关键字，`Character` 是类名 |
| `pass` | "这个类是空的，先占个位置。" Python 要求类体不能完全为空 |
| `hero = Character()` | 用类创建一个对象（实例化）。`Character()` 调用了 `__init__`（即使你没定义，Python 也有一个默认的） |
| `hero is monster` | 同一张蓝图，但建出来的是两栋不同的房子 |

### 3.2.2 类名命名规范

```python
# ✅ 大驼峰（每个单词首字母大写）
MyNetwork
LinearLayer
DataLoader

# ✅ 变量和函数用小写下划线
my_network
linear_layer
data_loader
```

### 3.2.3 一张图理解类和对象的关系

```
        ┌──────────────────────────┐
        │     class Character      │  ← 蓝图（只定义一次）
        │                         │
        │  - name                  │
        │  - hp                    │
        │  - attack                │
        │  - attack_enemy()        │
        └──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
  ┌───────────┐        ┌───────────┐
  │ hero      │        │ monster   │  ← 实例（可以创建任意多个）
  │ name="战士"│        │ name="哥布林"│
  │ hp=100    │        │ hp=50     │
  │ attack=20 │        │ attack=10 │
  └───────────┘        └───────────┘

  同一个蓝图，但它们是两个独立的对象。
  hero 的 hp 变化不影响 monster 的 hp。
```

---

## 3.3 __init__ 和 self —— 最核心的两个概念

### 3.3.1 __init__ 是什么时候被调用的

```python
# %%
class Character:
    def __init__(self, name, hp, attack):
        print(f"__init__ 被调用了！正在创建：{name}")
        self.name = name
        self.hp = hp
        self.attack = attack

# __init__ 在 Character(...) 的时候自动被调用
hero = Character("战士", 100, 20)
# 输出：__init__ 被调用了！正在创建：战士
```

**关键理解**：你不需要手动调用 `hero.__init__()`。Python 在执行 `Character("战士", 100, 20)` 时自动做了三件事：

```
Character("战士", 100, 20)
    ↓
1. 创建一个空的 Character 对象
2. 调用这个对象的 __init__(self, "战士", 100, 20)
3. 返回这个对象
```

### 3.3.2 self 到底是什么——最重要的概念

**self = "当前这个具体的对象本身。"**

```python
# %%
class Character:
    def __init__(self, name, hp):
        self.name = name     # self.name = 这个对象的 name 属性
        self.hp = hp         # self.hp = 这个对象的 hp 属性
    
    def who_am_i(self):
        print(f"我是 {self.name}，我有 {self.hp} 点生命")

hero = Character("战士", 100)
monster = Character("哥布林", 30)

hero.who_am_i()      # 我是 战士，我有 100 点生命
monster.who_am_i()   # 我是 哥布林，我有 30 点生命
```

**同一个 `self`，不同的指向**：

```
hero.who_am_i()    → self = hero    → self.name = "战士"
monster.who_am_i() → self = monster → self.name = "哥布林"
```

**self 的三个特征：**

| 特征 | 解释 |
|------|------|
| 自动传入 | 你写 `hero.method(x)`，Python 把它变成 `Character.method(hero, x)` |
| 指向当前对象 | `hero.method()` 中 self=hero；`monster.method()` 中 self=monster |
| 不是关键字 | 理论上可以叫别的名（如 `this`），但全 Python 世界都用 `self` |

### 3.3.3 实例属性 vs 类属性

```python
# %%
class Character:
    species = "人类"   # ← 类属性：所有角色共享
    
    def __init__(self, name, hp):
        self.name = name   # ← 实例属性：每个角色各自拥有
        self.hp = hp

hero = Character("战士", 100)
monster = Character("哥布林", 30)

# 类属性是所有实例共享的
print(hero.species)      # "人类"
print(monster.species)   # "人类"

# 实例属性是各自独立的
print(hero.name)         # "战士"
print(monster.name)      # "哥布林"

# 修改类属性会影响所有实例
Character.species = "未知生物"
print(hero.species)      # "未知生物"
print(monster.species)   # "未知生物"
```

### 3.3.4 用图理解 self.xxx

```
当你写 self.name = name 时：

    hero 对象：                      monster 对象：
    ┌──────────────────┐            ┌──────────────────┐
    │ self → hero      │            │ self → monster   │
    │ self.name = "战士"│            │ self.name = "哥布林"│
    │ self.hp = 100    │            │ self.hp = 30     │
    └──────────────────┘            └──────────────────┘

    self 就像一个指针，指向"当前正在操作的对象"。
    在 hero 里面，self 指向 hero。
    在 monster 里面，self 指向 monster。
```

---

## 3.4 方法 —— 类里面的函数

### 3.4.1 定义和使用方法

```python
# %%
class Character:
    def __init__(self, name, hp, attack):
        self.name = name
        self.hp = hp
        self.attack = attack
    
    def attack_enemy(self, enemy):
        """攻击另一个角色"""
        enemy.hp -= self.attack
        print(f"{self.name} 攻击了 {enemy.name}！")
        print(f"{enemy.name} 剩余 HP: {enemy.hp}")
    
    def is_alive(self):
        """是否还活着"""
        return self.hp > 0
    
    def heal(self, amount):
        """回复生命"""
        self.hp += amount
        print(f"{self.name} 恢复了 {amount} HP，当前 HP: {self.hp}")

# 使用
hero = Character("战士", 100, 25)
goblin = Character("哥布林", 50, 10)

hero.attack_enemy(goblin)     # 战士攻击哥布林
print(goblin.is_alive())       # True（还有 25 HP）
goblin.heal(20)                # 哥布林回复
```

### 3.4.2 方法 vs 普通函数

```
┌──────────────┬─────────────────────┬──────────────────────┐
│              │ 函数（function）      │ 方法（method）        │
├──────────────┼─────────────────────┼──────────────────────┤
│ 定义位置      │ 独立                 │ 类内部                │
│ 第一个参数    │ 任意                 │ 必须是 self          │
│ 调用方式      │ func(arg)           │ obj.method(arg)      │
│ self         │ 没有                 │ 自动传入              │
│ 访问对象数据  │ 需要传参              │ 通过 self 直接访问    │
└──────────────┴─────────────────────┴──────────────────────┘
```

### 3.4.3 方法调用的本质

```python
# %%
class Demo:
    def show(self, message):
        print(f"show: {message}")

d = Demo()

# 这两种写法效果完全一样：
d.show("hello")              # ✅ 正常写法
Demo.show(d, "hello")        # 效果相同，self 显式传入

# d.show("hello") 被 Python 自动变成 Demo.show(d, "hello")
```

---

## 3.5 完整示例：从函数到类——重构"计数器"

### 3.5.1 函数版本

```python
# %%
# 函数版本：数据和操作分离
count = 0

def increment(c):
    return c + 1

def decrement(c):
    return c - 1

def reset():
    return 0

# 使用——很别扭
count = increment(count)
count = increment(count)
count = increment(count)
print(f"计数: {count}")   # 3
count = reset()
print(f"重置: {count}")   # 0
```

**问题**：
- 如果你需要两个计数器（比如数苹果和数橘子），需要两套变量
- `count` 是全局变量，任何地方都能改

### 3.5.2 类版本

```python
# %%
class Counter:
    def __init__(self):
        self.value = 0          # 计数器的"状态"
    
    def increment(self):
        self.value += 1         # 操作自己的状态
        return self.value
    
    def decrement(self):
        self.value -= 1
        return self.value
    
    def reset(self):
        self.value = 0
        return self.value
    
    def get_value(self):
        return self.value

# 使用——很自然
apple_counter = Counter()
orange_counter = Counter()

apple_counter.increment()
apple_counter.increment()
orange_counter.increment()

print(f"苹果: {apple_counter.get_value()}")    # 2
print(f"橘子: {orange_counter.get_value()}")   # 1
# 两个计数器互不影响！
```

**为什么类版本更好：**

```
1. 封装：数据和操作打包在一起，不会散落各处
2. 独立：可以创建任意多个互不干扰的计数器
3. 安全：不能直接 apple_counter.value = -999（虽然没有强制阻止，但约定如此）
4. 清晰：apple_counter.increment() 比 count = increment(count) 更易读
```

---

## 3.6 继承 —— nn.Module 的基石

### 3.6.1 为什么需要继承

你写好了 `Counter` 类。现在你需要一个"有上限的计数器"——计数到 100 就归零。

你有两个选择：
- **A）重写一个新的类**：复制粘贴大部分代码 → 大量重复
- **B）继承 Counter**：只写和 Counter 不一样的部分

继承 = **在已有类的基础上，添加或修改功能。**

### 3.6.2 父类与子类

```python
# %%
class Counter:                        # ← 父类（基类）
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value

class BoundedCounter(Counter):        # ← 子类（派生类）
    # 只重写需要改的方法
    def increment(self):
        self.value += 1
        if self.value > 100:
            self.value = 0
        return self.value

# BoundedCounter 自动继承了 Counter 的其他一切
bc = BoundedCounter()
bc.increment()
print(bc.value)  # 1 （成功继承了 __init__）
```

**语法**：`class 子类(父类):`

### 3.6.3 现实世界类比

```
父类 = "家用电器"（定义了通电、开关等基本能力）
子类 = "洗衣机"（继承了家电的所有基本能力，加了洗衣、脱水）

父类 = "交通工具"（定义了速度、载重等基本属性）
子类 = "自行车"（继承了交通工具的属性，加了脚踏、刹车）

在 PyTorch 中：
父类 = nn.Module（PyTorch 写好的神经网络基类）
子类 = MyNetwork（你写的具体网络——继承了所有基础设施）
```

### 3.6.4 第一个继承示例——输入/输出

```python
# %%
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "..."   # 父类不定义具体叫声

class Dog(Animal):      # Dog 继承 Animal
    def speak(self):    # 重写 speak
        return "汪！汪汪！"

class Cat(Animal):      # Cat 继承 Animal
    def speak(self):    # 重写 speak
        return "喵～"

class Fish(Animal):     # 不重写 speak
    pass

dog = Dog("旺财")
cat = Cat("咪咪")
fish = Fish("尼莫")

print(f"{dog.name}: {dog.speak()}")    # 旺财: 汪！汪汪！
print(f"{cat.name}: {cat.speak()}")    # 咪咪: 喵～
print(f"{fish.name}: {fish.speak()}")  # 尼莫: ...（用的父类 speak）
```

**注意**：`Dog` 和 `Cat` 没有定义 `__init__`！它们直接**继承**了 `Animal.__init__`。所以 `Dog("旺财")` 实际上调用的是 `Animal.__init__`。

---

## 3.7 super() —— 调用父类的方法

### 3.7.1 什么时候需要 super()

当子类需要自己的 `__init__` 时，它通常还需要调用父类的 `__init__`：

```python
# %%
class Animal:
    def __init__(self, name):
        self.name = name
        print(f"Animal.__init__: {name}")

class Dog(Animal):
    def __init__(self, name, breed):
        # ❌ 如果不调用 super().__init__()：
        #    self.name 永远不会被设置！
        #    Animal 的 __init__ 被跳过了！
        
        # ✅ 先调用父类的 __init__：
        super().__init__(name)    # 这行调用 Animal.__init__(self, name)
        self.breed = breed        # 然后设置 Dog 特有的属性
        print(f"Dog.__init__: {name}, 品种={breed}")

dog = Dog("旺财", "金毛")
print(f"名字: {dog.name}")     # 旺财（来自 Animal.__init__）
print(f"品种: {dog.breed}")    # 金毛（来自 Dog.__init__）
```

**输出**：
```
Animal.__init__: 旺财
Dog.__init__: 旺财, 品种=金毛
```

### 3.7.2 super().__init__() 到底做了什么

```
super().__init__(name)
    ↓
"找到我的父类（Animal），调用它的 __init__ 方法"
    ↓
Animal.__init__(self, name)   ← 注意：self 还是 Dog 实例
    ↓
self.name = name   ← 这行代码在 Animal.__init__ 中执行
                   ← 但 self 是狗，所以是给狗设置了 name
```

### 3.7.3 不写 super() 的后果

```python
# %%
class Animal:
    def __init__(self, name):
        self.name = name

class BadDog(Animal):
    def __init__(self, name, breed):
        self.breed = breed      # 只设置了 breed
        # 忘记了 super().__init__(name)！

bad = BadDog("旺财", "金毛")
print(bad.breed)    # "金毛"  ✅
# print(bad.name)   # AttributeError! ❌ name 没有设置！
```

### 3.7.4 对应到 PyTorch

```python
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()   # ← 这行调用了 nn.Module.__init__()
        # 它做了很多必须的初始化工作：
        # - 设置 training = True
        # - 准备参数管理
        # - 等等

# 如果没有 super().__init__()：
# nn.Module 的内部机制不会启动
# parameters()、to()、train()、eval() 等方法都会失效
```

---

## 3.8 方法重写（override）

### 3.8.1 什么是重写

子类定义一个和父类**同名**的方法 → 子类的方法"覆盖"父类的。

```python
# %%
class Animal:
    def speak(self):
        return "动物叫声"   # 默认实现

class Duck(Animal):
    def speak(self):        # 和父类同名 → 重写
        return "嘎嘎嘎嘎"

class SilentFish(Animal):
    pass                    # 不定义 speak → 沿用父类的

duck = Duck()
fish = SilentFish()

print(duck.speak())   # "嘎嘎嘎嘎" — 用子类的
print(fish.speak())   # "动物叫声" — 用父类的
```

### 3.8.2 forward() 就是一个被重写的方法

在 PyTorch 中：

```python
class MyNetwork(nn.Module):
    def forward(self, x):    # ← 重写了 nn.Module 的 forward
        return self.layer(x)

# nn.Module 原本的 forward 是空的（什么都不做）
# 你重写它来定义你自己的前向传播逻辑
# 这就是为什么方法必须叫 forward——PyTorch 在 model(x) 时会调用它
```

---

## 3.9 把 OOP 概念映射到 PyTorch

```
    Python OOP 概念               PyTorch 对应

    ┌──────────────────┐        ┌───────────────────────┐
    │ class             │  →    │ class MyNet(nn.Module):│
    │ 继承 (class A(B)) │  →    │ 继承自 nn.Module       │
    │ __init__          │  →    │ 定义 self.fc1, self.fc2│
    │ super().__init__()│  →    │ 初始化 nn.Module 部分   │
    │ 方法重写           │  →    │ def forward(self, x):  │
    │ self              │  →    │ 指向当前网络实例         │
    │ 实例属性           │  →    │ self.fc1 = nn.Linear()│
    │ 方法调用           │  →    │ model(x) → __call__   │
    │                   │       │        → forward(x)    │
    └──────────────────┘        └───────────────────────┘
```

**你写的每一个神经网络都是这样：**

```python
class MyNetwork(nn.Module):          # ← 继承：拿到 nn.Module 的所有能力
    def __init__(self):              # ← 初始化：定义网络有什么层
        super().__init__()           # ← 激活 nn.Module 的内部机制
        self.fc1 = nn.Linear(10, 5)  # ← 实例属性：网络的第一层
        self.fc2 = nn.Linear(5, 2)   # ← 实例属性：网络的第二层
    
    def forward(self, x):            # ← 方法重写：定义数据怎么流动
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

**如果第三章的内容你都理解了，这个代码的每一行就不再是"固定写法"——你知道每一行为什么存在。**

---

## 3.10 本章总结

```
    ┌───────────────┐
    │  class        │  蓝图，定义"有什么数据+能做什么"
    │  object       │  按蓝图造出来的具体东西
    │  __init__     │  创建对象时自动调用，设置初始状态
    │  self         │  "当前这个对象"，指向调用方法的那个实例
    │  方法         │  类里面的函数，第一个参数永远是 self
    │  继承         │  class 子类(父类): 复用父类的代码
    │  super()      │  调用父类的方法（特别是 __init__）
    │  方法重写     │  子类定义同名方法，覆盖父类的行为
    │  instance attr│  self.xxx = ... 每个对象独立的属性
    └───────────────┘
```

---

## 3.11 本章练习

### 练习 3-1：写一个"学生"类

```python
# 创建 Student 类：
# - __init__(self, name, score)
# - report(self): 打印 "{name} 的分数是 {score}"
# - is_passed(self): score >= 60 返回 True
# 创建 3 个学生，分别调用 report 和 is_passed
```

### 练习 3-2：写一个"银行账户"类

```python
# 创建 BankAccount 类：
# - __init__(self, owner, balance=0)
# - deposit(self, amount): 存钱，返回新余额
# - withdraw(self, amount): 取钱，不足时打印警告并返回 False
# - get_balance(self): 返回余额
```

### 练习 3-3：继承练习

```python
# 从 BankAccount 继承出 SavingsAccount：
# - 新增属性 interest_rate（利率，如 0.03 表示 3%）
# - 新增方法 add_interest(self): 按利率增加余额
# - 正确调用 super().__init__()
```

### 练习 3-4：super() 调用顺序

```python
# 不运行代码，写出输出：
class A:
    def __init__(self):
        print("A init")

class B(A):
    def __init__(self):
        print("B init")
        super().__init__()

class C(A):
    def __init__(self):
        super().__init__()
        print("C init")

b = B()
print("---")
c = C()
```

### 练习 3-5：写一个"计算器"类（链式调用）

```python
# 创建 Calculator 类：
# - __init__: result = 0
# - add(n), subtract(n), multiply(n), divide(n)
# - reset(), get_result()
# 要求：每个运算方法返回 self，支持链式调用
# 示例：calc.add(5).multiply(2).subtract(3).get_result() → 7
```

### 练习 3-6：不看答案——独立写"图书馆"系统

> 关闭所有文档，独立完成：

```python
# 写两个类：
# Book: title, author, is_borrowed(默认False), borrow(), return_book()
# Library: name, books(列表), add_book(), borrow_book(title), 
#          return_book(title), list_available()
# 创建至少 3 本书和 1 个图书馆，完成完整的借书→归还→列出流程
```

### 练习 3-7：代码重构

```python
# 以下是用散装函数写的银行系统。请重构为一个 Bank 类。

accounts = {}

def create_account(name, balance=0):
    accounts[name] = balance

def deposit(name, amount):
    if name in accounts:
        accounts[name] += amount

def withdraw(name, amount):
    if name in accounts and accounts[name] >= amount:
        accounts[name] -= amount

# 重构要求：
# 1. 所有功能封装在 Bank 类中
# 2. 可以创建多个互不干扰的 Bank 实例
# 3. 用方法替代函数
```

### 练习 3-8：对照练习——从 OOP 到 nn.Module

```python
# 不看第四章，只用第三章的知识 + 你的直觉，尝试解释：
#
# class MyNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(3, 2)
#     
#     def forward(self, x):
#         return self.fc(x)
#
# 1. nn.Module 为什么要放在括号里？           → （提示：3.6）
# 2. super().__init__() 做了什么？            → （提示：3.7）
# 3. self.fc 为什么用 self？                 → （提示：3.3）
# 4. forward 为什么没有调用语句？             → （提示：3.8）
```

---

## 答案与提示

<details>
<summary>练习 3-4 答案</summary>

```
B init
A init
---
A init
C init

原因：
B(): 先进入 B.__init__ → 打印"B init" → super().__init__() → 进入 A.__init__ → 打印"A init"
C(): 先进入 C.__init__ → super().__init__() 先执行 → 进入 A.__init__ → 打印"A init" → 回到 C.__init__ → 打印"C init"
```
</details>

<details>
<summary>练习 3-5 提示</summary>

```python
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, n):
        self.result += n
        return self     # 返回 self，链式调用的关键！
    
    def get_result(self):
        return self.result
```
</details>

<details>
<summary>练习 3-8 参考</summary>

```python
# 1. nn.Module 在括号里 → 继承（3.6）
#    MyNet 继承了 nn.Module 的所有能力（参数管理、设备迁移等）

# 2. super().__init__() → 调用父类的初始化（3.7）
#    激活 nn.Module 的内部机制（参数扫描、training 模式等）

# 3. self.fc → 实例属性（3.3）
#    nn.Module 通过扫描 self.xxx 来发现子模块
#    局部变量 fc = nn.Linear() 不会被发现

# 4. forward 不需要显式调用（3.8 + 3.4.3）
#    model(x) → model.__call__(x) → model.forward(x)
#    PyTorch 在 __call__ 中做了准备工作后自动调用 forward
```
</details>

---

> **下一步**：类、self、继承、super() 都理解了？现在进入[第四章：nn.Module](04_nn_module.md)，你会看到这些概念如何直接应用。
