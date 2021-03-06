# 算法复习

### 1. Sliding window

（1）给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字母的最小子串。

```python
示例：
输入: S = "ADOBECODEBANC", T = "ABC"
输出: "BANC"
```

```c++
string minWindow(string s, string t) 
{
    vector<int> needs(100,0);
    int length=s.size();
    if (length==0)
        return "";
    int conditions=0;   //记录有多少个unique的字符条件需要满足，如'ABCA'就有3个
    int length_t=t.size();
    if (length_t==0)
        return "";
    for (int i=0;i<length_t;i++)
    {
        needs[t[i]-'0']++;
        if (needs[t[i]-'0']==1)
            conditions++;
    }
    int left=0,right=0;   //双指针法
    while (right<length)
    {
        needs[s[right]-'0']--;
        if (needs[s[right]-'0']==0)
            conditions--;
        if (conditions==0)
            break;
        right++;
    } //先找到初始满足覆盖所有t的最短字符串，然后滑窗缩减
    if (right==length)
        return "";
    int answer[2]={left,right};
    int minLength=right-left;
    while (true)
    {
        while (conditions==0)
        {
            if ((right-left)<minLength)
            {
                minLength=right-left;
                answer[0]=left;
                answer[1]=right;
            }
            needs[s[left]-'0']++;
            if (needs[s[left]-'0']>0)
                conditions++;
            left++;
        }
        right++;
        if (right==length)
            break;
        needs[s[right]-'0']--;
        if (needs[s[right]-'0']==0)
            conditions--;
    }
    string result="";
    for (int i=answer[0];i<=answer[1];i++)
        result+=s[i];
    return result;
}
```

（2）给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 k 次。在执行上述操作后，找到包含重复字母的最长子串的长度。

注意:
字符串长度 和 k 不会超过 10^4。

```
示例：
输入:s = "ABAB", k = 2
输出:4
```

```python
def characterReplacement(self, s: str, k: int) -> int:
    # 用字典保存字母出现的次数，需要替换的字符数目＝窗口字符数目－数量最多的字符数目
    letter_num = {}
    l = 0
    res = 0
    for r in range(len(s)):
        letter_num[s[r]] = letter_num.get(s[r], 0) + 1
        max_letter = max(letter_num, key=letter_num.get)
        # 如果替换的字符数目超过给定的k，则移动左边界
        while r - l + 1 - letter_num[max_letter] > k:
            letter_num[s[l]] -= 1
            l += 1
            # 需要更新最多个数的字符
            max_letter = max(letter_num, key=letter_num.get)
        # 如果s[r]　超出了替换的字符数目，需要先处理，再计算结果
        res = max(res, r - l + 1)

    return res
```



### 2. 递归

（1）给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。例如，从根到叶子节点路径 1->2->3 代表数字 123。计算从根到叶子节点生成的所有数字之和。

```
示例：
输入: [1,2,3]
    1
   / \
  2   3
输出: 25
解释:从根到叶子节点路径 1->2 代表数字 12.从根到叶子节点路径 1->3 代表数字 13.因此，数字总和 = 12 + 13 = 25.
```

```c++
int answer=0;

int sumNumbers(TreeNode* root) 
{
    if (root==NULL)
        return 0;
    getsum(root,root->val);
    return answer;
}

void getsum(TreeNode* root,int val)
{
    if ((root->left==NULL)&&(root->right==NULL))
    {
        answer+=val;
        return ;
    }
    if (root->left!=NULL)
        getsum(root->left,val*10+root->left->val);
    if (root->right!=NULL)
        getsum(root->right,val*10+root->right->val);
}
```

（2）找到给定字符串（由小写字符组成）中的最长子串 **T** ， 要求 **T** 中的每一字符出现次数都不少于 *k* 。输出 **T** 的长度。

```
示例：
输入:
s = "ababbc", k = 2
输出:
5
最长子串为 "ababb" ，其中 'a' 重复了 2 次， 'b' 重复了 3 次。
```

```java
int longestSubstring(String s, int k) 
{
    int len = s.length();
    if (len == 0 || k > len) return 0;
    if (k < 2) return len;

    return count(s.toCharArray(), k, 0, len - 1);
}

int count(char[] chars, int k, int p1, int p2) 
{
    if (p2 - p1 + 1 < k) return 0;
    int[] times = new int[26];  //  26个字母,统计出现频次
    for (int i = p1; i <= p2; ++i) 
        ++times[chars[i] - 'a'];
    //  如果该字符出现频次小于k，则不可能出现在结果子串中.分别排除，然后挪动两个指针
    while (p2 - p1 + 1 >= k && times[chars[p1] - 'a'] < k)    ++p1;
    while (p2 - p1 + 1 >= k && times[chars[p2] - 'a'] < k)    --p2;

    if (p2 - p1 + 1 < k) 
        return 0;
    //  得到临时子串，再递归处理
    for (int i = p1; i <= p2; ++i) 
        if (times[chars[i] - 'a'] < k) 
            //  如果第i个不符合要求，切分成左右两段分别递归求得
            return Math.max(count(chars, k, p1, i - 1), count(chars, k, i + 1, p2));
    return p2 - p1 + 1;
}
```

​	递归拆分子串，分治。先统计出每个字符出现的频次，维护一对双指针，从首尾开始统计，从首尾往中间排除，如果出现次数小于k则不可能出现在最终子串中，排除并挪动指针，然后得到临时子串，依次从头遍历，一旦发现出现频次小于k的字符，以该字符为分割线，分别递归求其最大值返回。



（3）根据一棵树的前序遍历与中序遍历构造二叉树。

**注意:**你可以假设树中没有重复的元素。

```
示例：
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：
    3
   / \
  9  20
    /  \
   15   7
```

```python
def buildTree(self, preorder, inorder):
    def helper(in_left = 0, in_right = len(inorder)):
        nonlocal pre_idx
        # if there is no elements to construct subtrees
        if in_left == in_right:
            return None

        # pick up pre_idx element as a root
        root_val = preorder[pre_idx]
        root = TreeNode(root_val)

        # root splits inorder list
        # into left and right subtrees
        index = idx_map[root_val]

        # recursion 
        pre_idx += 1
        # build left subtree
        root.left = helper(in_left, index)
        # build right subtree
        root.right = helper(index + 1, in_right)
        return root
        
    # start from first preorder element
    pre_idx = 0
    # build a hashmap value -> its index
    idx_map = {val:idx for idx, val in enumerate(inorder)} 
    # 中序遍历中，根节点将数组分为左子树和右子树
    return helper()
```

（4）给定一个整数数组  `nums` 和一个正整数 `k`，找出是否有可能把这个数组分成 `k` 个非空子集，其总和都相等。

```
示例：
输入： nums = [4, 3, 2, 3, 5, 2, 1], k = 4
输出： True
说明： 有可能将其分成 4 个子集（5），（1,4），（2,3），（2,3）等于总和。
```

```java
boolean backtracking(int[] nums, int k, int target, int cur, 
                     int start, boolean[] used) 
{
    if (k == 0) return true;
    if (cur == target) 
        return backtracking(nums, k-1, target, 0, 0, used);
    for (int i = start; i < nums.length; i++) 
    {
        if (!used[i] && cur+nums[i] <= target) 
        {
            used[i] = true;
            if (backtracking(nums, k, target, cur+nums[i], i+1, used)) 
                return true;
            used[i] = false;
        }
    }
    return false;
}
    
boolean canPartitionKSubsets(int[] nums, int k) 
{
    int sum = 0, maxNum = 0;
    for (int i = 0; i < nums.length; i++) 
    {
        sum += nums[i];
        if (maxNum < nums[i]) 
            maxNum = nums[i];
    }
    if (sum % k != 0 || maxNum > sum/k) 
        return false;
    boolean[] used = new boolean[nums.length];
    return backtracking(nums, k, sum/k, 0, 0, used);
}
```



### 3. 动态规划

（1）给定一个字符串 *s*，将 *s* 分割成一些子串，使每个子串都是回文串。返回符合要求的最少分割次数。

```
示例：
输入: "aab"
输出: 1
```

```c++
int length;

int minCut(string s) 
{
    length=s.size();
    if (length==0)
        return -1;
    vector<int> dp(length,length-1);  // dp这里表示目前为止的mincut
    for (int i=0;i<length;i++)
    {
        this->cut(s,i,i,dp);   // 回文串考虑奇数长度和偶数长度两种
        this->cut(s,i,i+1,dp);
    }
    return dp[length-1];
}

void cut(const string s,int left,int right,vector<int>& dp)
{
    while ((left>=0)&&(right<length)&&(s[left]==s[right]))
    {
        dp[right]=min(dp[right],(left==0?-1:dp[left-1])+1);
        left--;
        right++;
    }
}
```

（2）给定一个整数数组 `nums` ，找出一个序列中乘积最大的连续子序列（该序列至少包含一个数）

```
示例：
输入: [-2,0,-1]
输出: 0
```

```c++
int maxProduct(vector<int>& nums) 
{
    int curr_min=1,curr_max=1;
    int answer=INT_MIN;
    for (int i=0;i<nums.size();i++)
    {
        if (nums[i]<0)
        {
            int tmp=curr_min;
            curr_min=curr_max;
            curr_max=tmp;
        }
        curr_min=min(curr_min*nums[i],nums[i]);
        curr_max=max(curr_max*nums[i],nums[i]);

        answer=max(answer,curr_max);
    }
    return answer;
}
```

（3）给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1.

```
示例：
输入: coins = [1, 2, 5], amount = 11
输出: 3 
```

```c++
int coinChange(vector<int>& coins, int amount) 
{
    vector<int> dp(amount+1,0);
    int result= get_num(coins,amount,dp);
    return result;
}
    
int get_num(vector<int>& coins, int amount, vector<int>& dp)
{
    if (amount==0)
        return 0;
    if (amount<0)
        return -1;
    if (dp[amount]!=0)
        return dp[amount];

    int min_num=INT_MAX;
    for (int i=0;i<coins.size();i++)
    {
        int result=get_num(coins,amount-coins[i],dp);
        if (result>=0)
            min_num=min(min_num,result+1);
    }

    dp[amount]=(min_num<INT_MAX)?min_num:-1;
    return dp[amount];
}
```

（4）给定一个整型数组, 你的任务是找到所有该数组的递增子序列，递增子序列的长度至少是2。

```
示例：
输入: [4, 6, 7, 7]
输出: [[4, 6], [4, 7], [4, 6, 7], [4, 6, 7, 7], [6, 7], [6, 7, 7], [7,7], [4,7,7]]
```

说明：

1. 给定数组的长度不会超过15。
2. 数组中的整数范围是 [-100,100]。
3. 给定数组中可能包含重复数字，相等的数字应该被视为递增的一种情况。

```c++
unordered_map<int, vector<vector<int>>> longest; 
void dfs(vector<int>& nums, int index)
{
    int num = nums[index];
    bool same = false;
    for(int i=index+1;i<nums.size();++i)
    { //从当前下标向后扫描，因为后面的下标都已经记录到map中了，
      //直接取序列的集合并在每个序列头部插入当前的数字
        if(num <= nums[i]) 
        {
            for(auto v: longest[i]) 
            {
                v.insert(v.begin(), num);
                longest[index].push_back(v);
            } 
            if(num == nums[i]) 
            { //遇到相同的数，后面的都不需要再加入map了,
              //因为后面这个数起始的递增序列已经计算过了。
                same = true;
                break;
            }
        } 
    }
    if(!same) 
        longest[index].push_back({num}); 
    	// 如果后面没有发现相同的数，说明这个数字单独构成的序列还没有加入，需要存入map.
}
    
vector<vector<int>> findSubsequences(vector<int>& nums) 
{
    if(nums.size() < 1) return {};
    longest[nums.size()-1] = {{nums[nums.size()-1]}};
    for(int i=nums.size()-2;i>=0;--i)
        dfs(nums, i);
    vector<vector<int>> result;
    for(int i=0;i<nums.size()-1;++i)
        for(auto v: longest[i])
            if(v.size() > 1) 
                result.push_back(std::move(v));   

    return result;
}
```

（5）给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```
示例：
输入: [3,3,5,0,0,3,1,4]
输出: 6
输入: [7,6,4,3,1] 
输出: 0 
```

```python
def maxProfit(self, prices) -> int:
    l=len(prices)
    if l<2:
        return 0
    res=0
    #先算一个dp1数组,dp1[i]表示截止到第i-1天只进行一次买卖的最大利润
    dp1=[0 for i in range(l)]
    max_price,min_price=prices[0],prices[0]
    for i in range(1,l):
        dp1[i]=max(dp1[i-1],prices[i]-min_price)
        #对于第i天来说，1.如果当天卖：最大利润即当前卖出价格
        #减去之前的最小买入价格，2如果不卖：最大利润和前一天的最大利润相同
        min_price=min(min_price,prices[i])  #更新当前最小买入价格
    #对于任意k，dp1[k]表示k卖出的最大利润，那么需要求剩下k+1到n-1的最大利润
    #倒着求，因为右边界不变始终为l-1，左边界在变化
    #dp2[i]表示从i开始到最后只进行一次买卖的最大利润
    res=dp1[-1]
    dp2=[0 for i in range(l)]
    max_price=prices[-1]
    for i in range(l-2,-1,-1):
        dp2[i]=max(dp2[i+1],max_price-prices[i])
        #对于第i天，1.若当天买，则最大利润即之后的最大卖出价格减去
        #当前买入价格，2.若当天不买，最大利润和后一天的最大利润相同
        max_price=max(max_price,prices[i])  #更新当前最大卖出价格
        res=max(res,dp1[i-1]+dp2[i]) if i>=1 else max(res,dp2[i])

    return res
```

（6）**（完全背包问题）**给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。 

```
示例：
输入: amount = 5, coins = [1, 2, 5]
输出: 4
解释: 有四种方式可以凑成总金额:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
```

```python
def change(self, amount: int, coins: List[int]) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(amount - coin + 1):
            if dp[i]:
                dp[i + coin] += dp[i]
    return dp[amount]
```

（7）我们有两个长度相等且不为空的整型数组 A 和 B 。

我们可以交换 A[i] 和 B[i] 的元素。注意这两个元素在各自的序列中应该处于相同的位置。

在交换过一些元素之后，数组 A 和 B 都应该是严格递增的。

给定数组 A 和 B ，请返回使得两个数组均保持严格递增状态的最小交换次数。假设给定的输入总是有效的。

```
示例:
输入: A = [1,3,5,4], B = [1,2,3,7]
输出: 1
```

```python
'''
swap[i]第i位 转 的累加次数
not_swap[i]第i位 不转 的累加次数

首先，我们只是比较然后改变swap和not_swap，并不会对两个数组的值做任何交换改变。
那么，A,B第i位的大小比较，由于前i-1位有旋转的可能，导致比较不应该局限于A[i]和A[i-1], B[i]和B[i-1]之间，而应该存在交叉比较。
接下来，就列出了交叉比较的三种情况。

在第i位时，分三种情况：

1.if A[i-1] >= B[i] or B[i-1] >= A[i]:
    swap[i] = swap[i-1] + 1  # 要换，因为i-1换了，才会出现交叉比较的违规情况需要被交换，swap[i-1]+1
    not_swap[i] = not_swap[i-1]  # 不换，只能因为i-1没换，所以交叉比较的违规情况可以忽视
2.elif A[i-1] >= A[i] or B[i-1] >= B[i]:
    swap[i] = not_swap[i-1]+1  # 要换，因为前面没换，这里违规，就必须处理
    not_swap[i] = swap[i-1]  # 不换，因为i-1换了，这里的违规，可以忽略
3.elif A[i-1] < A[i] or B[i-1] < B[i]:
    这种情况下转不转都可以，为了保证次数最少，我们选i-1位转和不转的最小值
    swap[i] = min(swap[i-1], not_swap[i-1])+1 # 记住要+1，因为swap表示第i位要旋转
    not_swap[i] = min(swap[i-1], not_swap[i-1]) # 不用+1， 因为我们选择第i位不旋转
!!注意3.必须放在最后
'''
def minSwap(self, A, B):
    swap = [0]*len(A)
    not_swap = [0]*len(A)
    swap[0] = 1 # 由于循环从1开始，第0位如果旋转，swap[0] = 1
        
    for i in range(1, len(A)):
        if B[i-1] >= A[i] or A[i-1] >= B[i]:
            swap[i] = swap[i-1] + 1
            not_swap[i] = not_swap[i-1]
        elif A[i-1] >= A[i] or B[i-1] >= B[i]:
            swap[i] = not_swap[i-1] + 1
            not_swap[i] = swap[i-1]
        elif A[i-1] < A[i] or B[i-1] < B[i]:
            temp = min(swap[i-1], not_swap[i-1])
            not_swap[i] = temp
            swap[i] = temp + 1
    return min(swap[-1], not_swap[-1])
```

（8）给定一个未排序的整数数组，找到最长递增子序列的个数。

```
示例：
输入: [1,3,5,4,7]
输出: 2
解释: 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]。
```

```python
def findNumberOfLIS(self, nums):
    N = len(nums)
    if N <= 1: return N
    lengths = [0] * N #lengths[i] = longest ending in nums[i]
    counts = [1] * N #count[i] = number of longest ending in nums[i]

    for j, num in enumerate(nums):
        for i in xrange(j):
            if nums[i] < nums[j]:
                if lengths[i] >= lengths[j]:
                    lengths[j] = 1 + lengths[i]
                    counts[j] = counts[i]
                elif lengths[i] + 1 == lengths[j]:
                    counts[j] += counts[i]

    longest = max(lengths)
    return sum(c for i, c in enumerate(counts) if lengths[i] == longest)
```

· 假设对于以 `nums[i]` 结尾的序列，我们知道最长序列的长度 length[i]，以及具有该长度的序列的 count[i]。
对于每一个 i<j 和一个 A[i]<A[j]，我们可以将一个 A[j] 附加到以 A[i] 结尾的最长子序列上。
· 如果这些序列比 length[j] 长，那么我们就知道我们有count[i] 个长度为 length 的序列。如果这些序列的长度与 length[j] 相等，那么我们就知道现在有 count[i] 个额外的序列（即 count[j]+=count[i]）

### 4. 哈希表（字典）

（1）给定一个整数数组和一个整数 **k，**你需要找到该数组中和为 **k** 的连续的子数组的个数。

```
示例：
输入:nums = [1,1,1], k = 2
输出: 2
```

```c++
int subarraySum(vector<int>& nums, int k) 
{
    int length=nums.size();
    map<int,int> look_up;
    look_up[0]=1;
    int sum=0,answer=0;
    for (int i=0;i<length;i++)
    {
        sum+=nums[i];
        int delta=sum-k;
        if (look_up.find(delta)!=look_up.end())
        	answer+=look_up[delta];

        if (look_up.find(sum)!=look_up.end())
        	look_up[sum]++;
        else
        	look_up[sum]=1;
    }
    return answer;
}
```

（2）给定一个二进制数组, 找到含有相同数量的 0 和 1 的最长连续子数组（的长度），给定的二进制数组的长度不会超过50000.

```
输入: [0,1,0]
输出: 2
说明: [0, 1] (或 [1, 0]) 是具有相同数量0和1的最长连续子数组。
```

```c++
int findMaxLength(int[] nums) 
{
    Map<Integer, Integer> map = new HashMap<>();  
    //用map记录所有的(count,index)对，其中index只记录第一次出现
    map.put(0, -1);
    int maxlen = 0, count = 0;
    for (int i = 0; i < nums.length; i++) 
    {
        count = count + (nums[i] == 1 ? 1 : -1);
        if (map.containsKey(count))
            maxlen = Math.max(maxlen, i - map.get(count));
        else 
            map.put(count, i);
    }
    return maxlen;
}
```

![image.png](https://pic.leetcode-cn.com/ce11808babb2a9321a336f58d4f00f32d63d55adb7c7e79d0890e164c0e11691-image.png)



### 5. 二分法

（1）珂珂喜欢吃香蕉。这里有 `N` 堆香蕉，第 `i` 堆中有 `piles[i]` 根香蕉。警卫已经离开了，将在 `H` 小时后回来。

珂珂可以决定她吃香蕉的速度 K （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 K 根。如果这堆香蕉少于 K 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。  

珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。

返回她可以在 H 小时内吃掉所有香蕉的最小速度 K（K 为整数）。

- `1 <= piles.length <= 10^4`
- `piles.length <= H <= 10^9`
- `1 <= piles[i] <= 10^9`

```
示例：
输入: piles = [30,11,23,4,20], H = 6
输出: 23
```

```c++
int minEatingSpeed(vector<int>& piles, int H) 
{
	int lo = 1, hi = pow(10, 9);
	while (lo < hi) 
    {
        int mi = lo + (hi - lo) / 2;
        if (!possible(piles, H, mi))
            lo = mi + 1;
        else
            hi = mi;
	}
	return lo;
}

// Can Koko eat all bananas in H hours with eating speed K?
bool possible(vector<int>& piles, int H, int K) 
{
	int time = 0;
	for (int p: piles)
		time += (p - 1) / K + 1;
	return time <= H;
}
```



（2）给定两个大小为 m 和 n 的有序数组 `nums1` 和 `nums2`。

请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 `O(log(m + n))`。

你可以假设 `nums1` 和 `nums2` 不会同时为空。

```
示例：
nums1 = [1, 2]
nums2 = [3, 4]
则中位数是 (2 + 3)/2 = 2.5
```

```c++
double findMedianSortedArrays(int[] A, int[] B) 
{
    int m = A.length;
    int n = B.length;
    if (m > n) 
    {
        int[] temp = A; A = B; B = temp;
        int tmp = m; m = n; n = tmp;
    }
    int iMin = 0, iMax = m, halfLen = (m + n + 1) / 2;
    while (iMin <= iMax) 
    {
        int i = (iMin + iMax) / 2;
        int j = halfLen - i;
        if (i < iMax && B[j-1] > A[i])
            iMin = i + 1;    // i is too small
        else if (i > iMin && A[i-1] > B[j]) 
            iMax = i - 1;    // i is too big
        else                 // i is perfect
        { 
            int maxLeft = 0;
            if (i == 0) 
                maxLeft = B[j-1]; 
            else if (j == 0) 
                maxLeft = A[i-1];
            else  
                maxLeft = max(A[i-1], B[j-1]);
            
            if ((m + n) % 2 == 1)  
                return maxLeft;
            int minRight = 0;
            if (i == m)  
                minRight = B[j];
            else if (j == n) 
                minRight = A[i];
            else 
                minRight = min(B[j], A[i]);

            return (maxLeft + minRight) / 2.0;
        }
    }
    return 0.0;
}
```

(3）给定一个 *n x n* 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第k小的元素。1 ≤ k ≤ n^2

```
示例：
matrix = [   [ 1,  5,  9],   [10, 11, 13],   [12, 13, 15]],
k = 8,

返回：13
```

```java
int kthSmallest(int[][] matrix, int k) 
{
    int row = matrix.length;
    int col = matrix[0].length;
    int left = matrix[0][0];
    int right = matrix[row - 1][col - 1];
    while (left < right) 
    {
        int mid = (left + right) / 2;
        // 找二维矩阵中<=mid的元素总个数
        int count = findNotBiggerThanMid(matrix, mid, row, col);
        if (count < k)
            left = mid + 1;
        else
            right = mid;
    }
    return right;
}

int findNotBiggerThanMid(int[][] matrix, int mid, int row, int col) 
{
    // 以列为单位找，找到每一列最后一个<=mid的数即知道每一列有多少个数<=mid
    int i = row - 1;
    int j = 0;
    int count = 0;
    while (i >= 0 && j < col) 
    {
        if (matrix[i][j] <= mid) 
        {
            count += i + 1;  // 第j列有i+1个元素<=mid
            j++;
        } 
        else
            i--;
    }
    return count;
}
```



### 6. 排序

（1）在 *O*(*n* log *n*) 时间复杂度和常数级空间复杂度下，对链表进行排序

```
示例：
输入: 4->2->1->3
输出: 1->2->3->4
```

```c++
ListNode* sortList(ListNode* head) 
    return mergesort(head);

ListNode* mergesort(ListNode* node)
{
    if(!node || !node->next) return node;
    ListNode *fast=node;//快指针走两步
    ListNode *slow=node;//慢指针走一步
    ListNode *brek=node;//断点
    while(fast && fast->next)
    {
        fast=fast->next->next;
        brek=slow;
        slow=slow->next;
    }
    brek->next=nullptr;
    ListNode *l1=mergesort(node);
    ListNode *l2=mergesort(slow);
    return merge(l1,l2);
}

ListNode* merge(ListNode* l1,ListNode* l2)
{
    if(l1==NULL)
        return l2;
    if(l2==NULL)
        return l1;
    if(l1->val < l2->val)
    {
        l1->next=merge(l1->next,l2);
        return l1;
    }
    else
    {
        l2->next=merge(l2->next,l1);
        return l2;
    }
}
```

（2）给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```
例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，
满足要求的三元组集合为：
[  [-1, 0, 1],  [-1, -1, 2]]
```

```java
List<List<Integer>> threeSum(int[] nums) 
{
    List<List<Integer>> ans = new ArrayList();
    int len = nums.length;
    if(nums == null || len < 3) 
        return ans;
    Arrays.sort(nums); // 排序
    for (int i = 0; i < len ; i++) 
    {
        if(nums[i] > 0) break; // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
        if(i > 0 && nums[i] == nums[i-1]) continue; // 去重
        int L = i+1;
        int R = len-1;
        while(L < R)
        {
            int sum = nums[i] + nums[L] + nums[R];
            if(sum == 0)
            {
                ans.add(Arrays.asList(nums[i],nums[L],nums[R]));
                while (L<R && nums[L] == nums[L+1])    L++; // 去重
                while (L<R && nums[R] == nums[R-1])    R--; // 去重
                L++;
                R--;
            }
            else if (sum < 0) L++;
            else if (sum > 0) R--;
        }
    }        
    return ans;
}
```

（3）给定一个整数数组 nums，按要求返回一个新数组 counts。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。

```
示例：
输入: [5,2,6,1]
输出: [2,1,1,0] 
```

```c++
using pii = pair<int, int>; // <number, index>
vector<int> countSmaller(vector<int>& nums) 
{
    vector<pii> v;
    v.reserve(nums.size());
        
    for (int i = 0; i < nums.size(); ++i)
        v.emplace_back(nums[i], i);
        
    vector<int> res(v.size());
    merge_sort(v, 0, v.size(), res);
    return res;
}
    
void merge_sort(vector<pii>& nums, int lo, int hi, vector<int>& res) 
{
    if (hi - lo <= 1) return; // 元素个数 <= 1 终止。
    int mid = lo + (hi - lo >> 1);
    merge_sort(nums, lo, mid, res);
    merge_sort(nums, mid, hi, res);

    int right = mid;
        
    // 对于左半区间中的每个元素 left，统计右侧比它小的元素的个数
    for (int left = lo; left < mid; ++left) 
    {
        while (right != hi && nums[left] > nums[right]) 
            ++right;
        res[nums[left].second] += right - mid;
    }
        
    inplace_merge(nums.begin() + lo, nums.begin() + mid, nums.begin() + hi);
}
```



### 7. 栈

（1）给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram_area.png)

```
示例：
输入: [2,1,5,6,2,3]
输出: 10
```

```python
def largestRectangleArea(self, heights):
    stack = []
    heights = [0] + heights + [0]
    res = 0
    for i in range(len(heights)):
        while stack and heights[stack[-1]] > heights[i]:
            tmp = stack.pop()
            res = max(res, (i - stack[-1] - 1) * heights[tmp])
        stack.append(i)
    return res
```

（2）累加数是一个字符串，组成它的数字可以形成累加序列。

一个有效的累加序列必须至少包含 3 个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。

给定一个只包含数字 '0'-'9' 的字符串，编写一个算法来判断给定输入是否是累加数。

说明: 累加序列里的数不会以 0 开头，所以不会出现 1, 2, 03 或者 1, 02, 3 的情况。

```
示例：
输入: "199100199"
输出: true 
解释: 累加序列为: 1, 99, 100, 199。1 + 99 = 100, 99 + 100 = 199
```

```python
def isAdditiveNumber(self, num):
    res=[] #将每次符合条件的数压入栈中
    return dfs(num,0)

def dfs(num,count): #count 记录当前找到的数的个数
    if count>=3 and len(num)==0: #当数量不少于三个且字符串为空，返回True
        return True
    for i in range(1,len(num)+1):
        if i>1 and num[0]=="0": #去掉0开头的数，但一位数的时候可以是0， 如1.0.1
            continue

        if count<2: #栈中数量不足三个，直接压入栈中
            res.append(int(num[:i])) 
            if dfs(num[i:],count+1):  #继续判断剩余字符串
                return True

        else:    #当个数足够两个，那么就开始判断前两个数之和是否与当前数相等
            if res[-1]+res[-2]==int(num[:i]): 
                res.append(int(num[:i])) 
                if dfs(num[i:],count+1):
                    return True
        res.pop(-1)  # 恢复原栈
    
    return False
```

（3）给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

```
示例:
输入:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出: 6
```

```python
def leetcode84(self, heights):
    # Get the maximum area in a histogram given its heights
    stack = [-1]
    maxarea = 0
    for i in range(len(heights)):
        while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
            maxarea = max(maxarea, heights[stack.pop()] * (i - stack[-1] - 1))
        stack.append(i)

    while stack[-1] != -1:
        maxarea = max(maxarea, 
                      heights[stack.pop()] * (len(heights) - stack[-1] - 1))
    return maxarea


def maximalRectangle(self, matrix: List[List[str]]) -> int:
    if not matrix: return 0

    maxarea = 0
    dp = [0] * len(matrix[0])
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # matrix[i][j] == '1'则直方图上增加1，否则从此处开始为0.
            dp[j] = dp[j] + 1 if matrix[i][j] == '1' else 0

        # update maxarea with the maximum area from this row's histogram
        maxarea = max(maxarea, leetcode84(dp))
    return maxarea
```

（4）给定一个仅包含小写字母的字符串，去除字符串中重复的字母，使得每个字母只出现一次。需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）。

```
示例：
输入: "bcabc"
输出: "abc"
```

```java
String removeDuplicateLetters(String s) 
{
    if (s.length()==0)
        return  new String();
    int[] words = new int[26];//记录字母出现的次数，用来判断该串后面是否还有这个字母
    boolean[] isIn = new boolean[26];//用来判断这个字母是否已经进栈
    for (int i=0;i<s.length();i++)  //统计字母个数
        words[s.charAt(i)-'a']++;
    Stack<Character> stack = new Stack<>();
    for (int i=0;i<s.length();i++)
    {
        words[s.charAt(i)-'a']--;
        if (!isIn[s.charAt(i)-'a']) 
        {   //先判断栈中没有这个字母，如果有这个字母无需进行任何操作直接跳过即可
            while(!stack.isEmpty()) 
            {
                if (!stack.isEmpty() && (s.charAt(i) - 'a' < stack.peek() - 'a' 
                                     && words[stack.peek() - 'a'] > 0)) 
                {   //如果当前字母的字典序小于栈顶字母且栈顶字母在该串的后面仍存在
                    isIn[stack.peek() - 'a'] = false;//将栈顶字母设置为不存在栈中
                    stack.pop();//使其出栈
                }
                else
                    break;//处理完后直接跳出
            }
            stack.push(s.charAt(i));
            isIn[s.charAt(i)-'a'] = true;//标记栈中已经有该字母
        }
    }
    StringBuilder result = new StringBuilder();
    while(!stack.isEmpty())
        result.append(stack.pop());//将栈中字母添加
    result.reverse();//倒转字符串
    String last = result.toString();
    return last;
}
```



### 8. 快慢指针

给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

说明：

1. 不能更改原数组（假设数组是只读的）。
2. 只能使用额外的 O(1) 的空间。
3. 时间复杂度小于 O(n^2) 。
4. 数组中只有一个重复的数字，但它可能不止重复出现一次。

```
示例：
输入: [3,1,3,4,2]
输出: 3
```

```c++
int findDuplicate(vector<int>& nums) 
{
    int fast = 0, slow = 0;
    while(true)
    {
        fast = nums[nums[fast]];
        slow = nums[slow];
        if(fast == slow)
            break;
    }
    int finder = 0;
    while(true)
    {
        finder = nums[finder];
        slow = nums[slow];
        if(slow == finder)
            break;        
    }
    return slow;
}
```

​	slow和fast会在环中相遇，假设一些量：起点到环的入口长度为m，环的周长为c，在fast和slow相遇时slow走了n步，fast走了2n步，fast比slow多走了n步，而这n步全用在了在环里循环。
当fast和last相遇之后，我们设置第三个指针finder，它从起点开始和在fast和slow相遇处的slow同步前进，当finder和slow相遇时，就是在环的入口处相遇，也就是重复的那个数字相遇。

![image.png](https://pic.leetcode-cn.com/970cf34694dd893c64924e1559617f64ad6b5b272a81ac3de5836cb6fb42fed7-image.png)

​	解释：fast和slow相遇时，slow在环中行进的距离是n-m，其中n%c==0。这时我们让再让slow前进m步——也就是在环中走了n步了。而n%c==0即slow在环里面走的距离是环的周长的整数倍，就回到了环的入口了，而入口就是重复的数字。我们不知道起点到入口的长度m，所以弄个finder和slow一起走，他们必定会在入口处相遇。



### 9. 贪心算法

（1）输入一个按升序排序的整数数组（可能包含重复数字），你需要将它们分割成几个子序列，其中每个子序列至少包含三个连续整数。返回你是否能做出这样的分割？

```
示例：
输入: [1,2,3,3,4,4,5,5]
输出: True
```

```python
def isPossible(self, nums: List[int]) -> bool:
    counter = dict()
    for n in nums:
        counter[n] = counter.get(n, 0) + 1

    end = dict()
    for n in nums:
        if counter[n] == 0:
            continue

        counter[n] -= 1
        if end.get(n - 1, 0) > 0:
            # 添加到已有子序列的末尾
            end[n - 1] -= 1
            end[n] = end.get(n, 0) + 1
        elif counter.get(n + 1, 0) > 0 and counter.get(n + 2, 0) > 0:
            # 添加到子序列头部
            counter[n + 1] -= 1
            counter[n + 2] -= 1
            end[n + 2] = end.get(n + 2, 0) + 1
        else:
            return False
    return True
```

（2）老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。你需要按照以下要求，帮助老师给这些孩子分发糖果：

1. 每个孩子至少分配到 1 个糖果。
2. 相邻的孩子中，评分高的孩子必须获得更多的糖果。

那么这样下来，老师至少需要准备多少颗糖果呢？

```
示例：
输入: [1,2,2]
输出: 4
解释: 你可以分别给这三个孩子分发 1、2、1 颗糖果。
```

```python
def candy(self, ratings):
    s = 0
    n=len(ratings)
    s+=n
    tmp =[0]*n
    for i in range(1,n):
        if ratings[i]>ratings[i-1]:
            tmp[i] = tmp[i-1]+1
    for i in range(n-2,-1,-1):
        if ratings[i]>ratings[i+1]:
            tmp[i]=max(tmp[i],tmp[i+1]+1)
    s+=sum(tmp)
    return s
```

（3）累加数是一个字符串，组成它的数字可以形成累加序列。一个有效的累加序列必须至少包含 3 个数。除了最开始的两个数以外，字符串中的其他数都等于它之前两个数相加的和。

给定一个只包含数字 '0'-'9' 的字符串，编写一个算法来判断给定输入是否是累加数。

说明: 累加序列里的数不会以 0 开头，所以不会出现 1, 2, 03 或者 1, 02, 3 的情况。

```
示例：
输入: "199100199"
输出: true 
解释: 累加序列为: 1, 99, 100, 199。1 + 99 = 100, 99 + 100 = 199
```

```python
def str2int(s):  
    if len(s)>=2 and s.startswith('0'):  # 如果不合法返回-1
        return -1
    ret, w = 0, 0
    for i in s[::-1]:
        ret += int(i)*10**w
        w += 1
    return ret

def isAdditiveNumber(self, nums: str) -> bool:
    leng = len(nums)
    pos = []
    for i in range(leng):  # 查找可能的前两个字母的切分
        for j in range(i+1, leng):
            if leng-j>=max(j-i,i):
                pos.append([i,j])
    for i,j in pos:
        n1, n2 = str2int(nums[:i+1]), str2int(nums[i+1:j+1])
        if n1>=0 and n2>=0:
            p = j
            while p<leng:
                tmp = n1+n2
                if nums[p+1:].startswith(str(tmp)):
                    n1 = n2
                    n2 = tmp
                    p += len(str(tmp))
                    if p==leng-1:  # 刚好用完所有的数
                        return True
                else:
                    break
    return False
```



### 10. KMP字符串匹配

以图中的例子来说，在 i 处失配，那么主字符串和模式字符串的前边6位就是相同的。又因为模式字符串的前6位，它的前4位前缀和后4位后缀是相同的，所以我们推知主字符串i之前的4位和模式字符串开头的4位是相同的。就是图中的灰色部分。

![img](https://pic4.zhimg.com/80/v2-03a0d005badd0b8e7116d8d07947681c_hd.jpg)

PMT数组即为前后最大匹配数。将PMT数组向后偏移一位。我们把新得到的这个数组称为next数组。

![img](https://pic1.zhimg.com/80/v2-40b4885aace7b31499da9b90b7c46ed3_hd.jpg)

```c++
int KMP(char * t, char * p) 
{
	int i = 0; 
	int j = 0;

	while (i < strlen(t) && j < strlen(p))
	{
		if (j == -1 || t[i] == p[j]) 
		{
			i++;
            j++;
		}
	 	else 
            j = next[j];
    	}

    if (j == strlen(p))
       return i - j;
    else 
       return -1;
}
```

next数组求解：从模式字符串的第一位(注意，不包括第0位)开始对自身进行匹配运算。 在任一位置，能匹配的最长长度就是当前位置的next值。

![img](https://pic1.zhimg.com/80/v2-645f3ec49836d3c680869403e74f7934_hd.jpg)

![img](https://pic3.zhimg.com/80/v2-06477b79eadce2d7d22b4410b0d49aba_hd.jpg)

![img](https://pic1.zhimg.com/80/v2-8a1a205df5cad7ab2f07498484a54a89_hd.jpg)

```c++
void getNext(char * p, int * next)
{
	next[0] = -1;
	int i = 0, j = -1;

	while (i < strlen(p))
	{
		if (j == -1 || p[i] == p[j])
		{
			++i;
			++j;
			next[i] = j;
		}	
		else
			j = next[j];
	}
}
```



### 11. 堆

你有 `k` 个升序排列的整数数组。找到一个**最小**区间，使得 `k` 个列表中的每个列表至少有一个数包含在其中。

```
示例：
输入:[[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
输出: [20,24]
解释: 
列表 1：[4, 10, 15, 24, 26]，24 在区间 [20,24] 中。
列表 2：[0, 9, 12, 20]，20 在区间 [20,24] 中。
列表 3：[5, 18, 22, 30]，22 在区间 [20,24] 中。
```

```c++
struct ele 
{
    int value;
    int index;
    int arrayIndex;
    ele(int v=0, int a=0, int i=0): value(v), arrayIndex(a), index(i) {}
};

bool compare(const ele &a, const ele &b)
{
    return a.value > b.value;
}
    
vector<int> smallestRange(vector<vector<int>>& nums) 
{    
    // 优先队列(堆)数据类型:priority_queue<Type, Container, Functional>
    priority_queue<ele, vector<ele>, compare> que;  // #include <queue>
    int max = INT_MIN;
    long numsSize = nums.size();
    for (int i=0; i<numsSize; i++) 
    {
        int v = nums[i][0];
        if (max < v) 
            max = v;
        que.push(ele(v, i, 0));
    }

    int rl = -100005, rr = 100005;
    while (true) 
    {
        ele e = que.top();
        que.pop();
        int arrayIndex = e.arrayIndex;
        int index = e.index;
        int l = e.value;
        int r = max;
        if (r - l < rr - rl) 
        {
            rr = r;
            rl = l;
        }
        auto array = nums[arrayIndex];
        if (index >= array.size())
            break;

        int v = array[index];
        if (max < v)
            max = v;
        que.push(ele(v, arrayIndex, index + 1));
    }
    return {rl, rr};
}
```

思路：
1、先考虑用一个数组A，将题目中提供的数组的第一位都保存进去。那么我们现在就可以得到一个区间范围，它的值是[A_min, A_max].
2、将数组A中最小值找到，将这个值替换成它所在数组的下一个值，这样我们就可以将A_min变大，就可能将区间缩小。当然这样做的时候，可能将A_max变大了，所以每次都需要重新检测A_min和A_max。当有其中的一个数组被遍历完的话，我们就可以找到最小区间。
3、因为每次遍历都需要从数组A中找最小值和它所属数组，所以我们可以建立一个结构体：值、值所属数组、值在所属数组的index。然后可以使用priority_queue来帮助我们完成寻找最小值的工作。
