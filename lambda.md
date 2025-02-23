### **Lambda Function:**
```python
lambda x: x * 3 if x % 2 != 0 else x
```

### **Flow of Execution (Step by Step)**

1️⃣ **Receives an input (`x`)**  
   - The lambda function starts by taking a number `x` as input.

2️⃣ **Checks if `x` is odd (`x % 2 != 0`)**  
   - The `%` (modulus) operator finds the remainder when `x` is divided by `2`.
   - If the remainder is **not** `0`, that means `x` is **odd**.

3️⃣ **Two possible outcomes based on the condition:**
   - **If `x` is odd (`x % 2 != 0` is `True`) → Multiply `x` by 3.**
   - **If `x` is even (`x % 2 != 0` is `False`) → Keep `x` unchanged.**

4️⃣ **Returns the result**  
   - The function **returns the final value** based on the condition.

---

### **Flowchart of Execution**
```
Start
  ↓
Receive x as input
  ↓
Check: Is x % 2 ≠ 0?
  ├── Yes → Multiply x by 3
  ├── No  → Keep x unchanged
  ↓
Return result
  ↓
End
```

---

### **Example Walkthrough with `map()`**
```python
numbers = [1, 2, 3, 4, 5]
ternary = lambda x: x * 3 if x % 2 != 0 else x 
print(list(map(ternary, numbers)))
```

#### **Step-by-Step Execution:**
| `x`  | `x % 2 != 0` | Condition Met? | **Result** |
|------|------------|---------------|------------|
| `1`  | `1 % 2 != 0 → True`  | Yes (Odd)  | `1 * 3 = 3` |
| `2`  | `2 % 2 != 0 → False` | No (Even)  | `2` (unchanged) |
| `3`  | `3 % 2 != 0 → True`  | Yes (Odd)  | `3 * 3 = 9` |
| `4`  | `4 % 2 != 0 → False` | No (Even)  | `4` (unchanged) |
| `5`  | `5 % 2 != 0 → True`  | Yes (Odd)  | `5 * 3 = 15` |

#### **Final Output:**
```
[3, 2, 9, 4, 15]
```

---

### **Summary of Execution:**
1️⃣ **Takes a number `x`**  
2️⃣ **Checks if it's odd or even (`x % 2 != 0`)**  
3️⃣ **If odd → Multiply by 3**  
4️⃣ **If even → Keep the same**  
5️⃣ **Returns the new value**  

---
