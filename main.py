from typing import List, Dict, Union, Literal
import streamlit as st

alphas = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Term:
    def __init__(self, coefficient: Union[int, float], variables=None):
        if variables is None:
            variables = {}
        self.coefficient: Union[int, float] = coefficient
        self.variables: Dict[str, int] = {i: 0 for i in alphas}
        self.variables.update(variables)

    def __repr__(self) -> str:
        if not any(self.variables.values()):
            return ("+" if self.coefficient >= 0 else "") + str(self.coefficient)
        return f"{'+' if self.coefficient > 0 else ''}{'-' if self.coefficient == -1 else self.coefficient if self.coefficient != 1 else ''}{''.join(map(lambda x: f'{x[0]}^' + '{' + f'{x[1]}' + '}' if x[1] != 1 else str(x[0]), sorted(filter(lambda x: x[1] != 0, self.variables.items()))))}"

    def __eq__(self, other: Union[int, float, "Term"]) -> bool:
        if other == 0:
            return self.coefficient == 0
        if isinstance(other, (int, float)):
            return False if self.coefficient != other or any(self.variables.values()) else True
        if isinstance(other, Term):
            return True if self.coefficient == other.coefficient and self.variables == other.variables \
                else False
        raise TypeError(f"{type(other)} 不能与 Term 比较")

    def __neg__(self) -> "Term":
        return Term(-self.coefficient, self.variables)

    def __copy__(self) -> "Term":
        return Term(self.coefficient, self.variables.copy())


class Expression:
    def __init__(self, terms: List[Term]):
        self.terms = terms
        self.clear()

    def __repr__(self) -> str:
        if not any(map(lambda x: x != 0, self.terms)):
            return "0"
        text = "".join(map(str, self.terms)).removeprefix("+")
        return text

    def clear(self) -> None:
        self.terms = list(filter(lambda x: x != 0, self.terms))

    @classmethod
    def simple_str(cls, string):
        if string == "":
            return Expression([Term(1)])
        term = Expression([Term(1)])
        for little in string.split("*"):
            neg = -1 if little[0] == "-" else 1
            if little[0] in "+-":
                little = little[1:]
            coefficient, left = Expression.till_num(little)
            # 获取字母
            vars = {alphas[_]: 0 for _ in range(26)}
            i = 0
            while i != len(left):
                if left[i] == "^":
                    be = left[i - 1]
                    num, left = Expression.till_num(left[i + 1:])
                    i = 0
                    vars[be] += num - 1
                    continue
                vars[left[i]] = vars[left[i]] + 1
                i += 1

            term *= Term(coefficient * neg, vars)
        return term

    @classmethod
    def till_num(cls, string: str) -> (float | int, str):
        if not string:
            return 1, ""
        if string[0] not in "-0123456789.":
            return 1, string
        for i in range(len(string)):
            if string[i] not in "-0123456789.":
                return float(string[:i]) if "." in string[:i] else int(string[:i]), string[i:]
        return float(string) if "." in string else int(string), ""

    @classmethod
    def from_str(cls, string: str):
        # 格式要求: +-加减*乘^幂 (*可省略) 5ab-8a^56de^4+()^
        # No.1 括号外拆项 (以括号外正负号分割)
        string = string.replace(" ", "")
        string = string.removeprefix("+")
        depth = 0
        dots, parts = [0], []
        neg = True if string[0] == "-" else False
        s = " " + string[1:] if neg else string
        for j, i in enumerate(s):
            if i == "(":
                depth += 1
            elif i == ")":
                depth -= 1
                assert depth >= 0
            elif i in "+-" and depth == 0 and (i == "+" or (j != 0 and s[j - 1] != "^")):
                dots.append(j)
        dots.append(len(string))
        for i in range(len(dots) - 1):
            parts.append(string[dots[i]:dots[i + 1]])
        # 首项添加符号
        if parts[0] and parts[0][0] != "-":
            parts[0] = "+" + parts[0]

        # No.2 计算简单项(无括号,无需再次计算)
        terms = Expression([])
        for i in filter(lambda x: "(" not in x, parts):
            terms += Expression.simple_str(i)

        # No.3 计算剩余项(有括号)
        for i in filter(lambda x: "(" in x, parts):
            # 找到括号端点
            dots = []
            depth = 0
            for j in range(len(i)):
                if i[j] == "(":
                    depth += 1
                    if i[j - 1] == "*":
                        j -= 1
                    if depth == 1:
                        dots.append([j])
                if i[j] == ")":
                    depth -= 1
                    if depth == 0:
                        dots[-1].append(j)
            parts = []
            for j in dots[::-1]:
                s = i[j[0]: j[1] + 1]
                s = s[1:] if s[0] == "*" else s
                s = s[1:] if s[0] == "(" else s
                s = s[:-1] if s[-1] == ")" else s
                parts.append(Expression.from_str(s))
                i = i[:j[0]] + i[j[1] + 1:]
            term = Expression.simple_str(i)
            for j in parts:
                term *= j
            terms += term
        terms.clear()
        return terms

    def __add__(self, other: Union[Term, "Expression", int, float]) -> "Expression":
        terms = self.__copy__()
        if isinstance(other, Term):
            for i in terms.terms:
                if i.variables == other.variables:
                    i.coefficient += other.coefficient
                    break
            else:
                terms.terms.append(other)
            terms.clear()
            return terms
        if isinstance(other, Expression):
            for i in other.terms:
                terms += i
            return terms
        if isinstance(other, Union[int, float]):
            return self + Term(other)
        raise TypeError(f"{type(self)}: {self} 不能与 {type(other)}: {other} 运算")

    def __radd__(self, other: Union[Term, "Expression"]) -> "Expression":
        return self + other

    def __neg__(self) -> "Expression":
        terms = list(map(lambda x: -x, self.terms.copy()))
        return Expression(terms)

    def __sub__(self, other: Union[Term, "Expression"]) -> "Expression":
        return self + -other

    def __rsub__(self, other: Union[Term, "Expression"]) -> "Expression":
        return -self + other

    def __mul__(self, other: Union[Term, "Expression", int, float]) -> "Expression":
        if isinstance(other, Term):
            terms = self.__copy__()
            for i in terms.terms:
                i.coefficient *= other.coefficient
                for j in other.variables.items():
                    i.variables[j[0]] = i.variables.get(j[0], 0) + j[1]
            terms.clear()
            return terms
        if isinstance(other, Expression):
            terms = Expression([])
            for i in self.terms:
                terms += other * i
            terms.clear()
            return terms
        if isinstance(other, Union[int, float]):
            terms = self.__copy__()
            for i in terms.terms:
                i.coefficient *= other
            return terms

    def __rmul__(self, other: Union[Term, "Expression", int, float]) -> "Expression":
        return self * other

    def __copy__(self) -> "Expression":
        return Expression([i.__copy__() for i in self.terms])


class Matrix:
    def __init__(self, value: List[List[Expression]]):
        self.value: List[List[Expression]] = value

    @property
    def width(self) -> int:
        return len(self.value[0])

    @property
    def height(self) -> int:
        return len(self.value)

    def __repr__(self) -> str:
        head = "\\begin{matrix}\n"
        content = "\\\\\n".join(map(lambda x: "&".join(map(str, x)), self.value))
        end = "\\\\\n\\end{matrix}"
        return f"{head}{content}{end}"

    def show(self, mode: Literal["p", "b", "B", "v", "V"] = "b") -> str:
        """ "p"：小括号边框
            "b"：中括号边框
            "B"：大括号边框
            "v"：单竖线边框
            "V"：双竖线边框"""
        return str(self).replace("matrix", f"{mode}matrix")

    def __add__(self, other: "Matrix") -> "Matrix":
        assert self.width == other.width and self.height == other.height, "相加矩阵大小不相等"
        temp = []
        for i in range(self.height):
            temp.append([self.value[i][j] + other.value[i][j] for j in range(self.width)])
        return Matrix(temp)

    def __neg__(self):
        temp = []
        for i in self.value:
            temp.append([])
            for j in i:
                temp[-1].append(-j)
        return Matrix(temp)

    def __sub__(self, other: "Matrix") -> "Matrix":
        return self + -other

    def __mul__(self, other: Union[int, float, Term, Expression, "Matrix"]) -> "Matrix":
        if isinstance(other, (int, float, Term, Expression)):
            temp = []
            for i in self.value:
                temp.append([])
                for j in i:
                    temp[-1].append(j * other)
            return Matrix(temp)
        if isinstance(other, Matrix):
            assert self.width == other.height, f"无法相乘{self.show()}, {other.show()}"
            temp = []
            for i in range(self.height):
                temp.append([sum(map(Expression.__mul__, self.value[i], map(lambda x: x[j], other.value))) for j in
                             range(other.width)])
            return Matrix(temp)

    def __copy__(self) -> "Matrix":
        temp = []
        for i in self.value:
            temp.append([])
            for j in i:
                temp[-1].append(j.__copy__())
        return Matrix(temp)

    def __pow__(self, power, modulo=None):
        assert self.width == self.height, "矩阵不是方阵"
        if power == -1:
            temp = []
            determinant = self.determinant()
            assert determinant != 0, "矩阵不存在逆矩阵"
            for i in range(self.width):
                temp.append([])
                for j in range(self.height):
                    temp[-1].append(self.remainder(i, j))
            return determinant, Matrix(temp)
        temp = self.__copy__()
        result = Matrix(
            [[Expression([Term(0)])] * i + [Expression([Term(1)])] + [Expression([Term(0)])] * (self.height - i - 1) for
             i in range(self.height)])
        for i in bin(power)[2:][::-1]:
            if i == "1":
                result = temp * result
            temp *= temp
        return result

    def remainder(self, row: int, column: int):
        assert 0 <= row < self.width and 0 <= column < self.height, "Error get remainder"
        temp = [i[:row] + i[row + 1:] for i in (self.value[:column] + self.value[column + 1:])]
        return solve(temp) * (-1 if (row + column) % 2 else 1)

    def determinant(self) -> Expression:
        assert self.width == self.height, f"该矩阵 ({self.width}x{self.height}) 无法计算行列式"
        return solve(self.value)

    def transpose(self) -> "Matrix":
        temp = []
        for i in range(self.width):
            temp.append([])
            for j in range(self.height):
                temp[-1].append(self.value[j][i].__copy__())
        return Matrix(temp)


def solve(m: List[List[Expression]]) -> Expression:
    n = len(m)
    if n == 1:
        return m[0][0]
    s = Expression([])
    for i in range(n):
        temp = [m[j][:i] + m[j][i + 1:] for j in range(1, n)]
        if i % 2 == 1:
            s -= solve(temp) * m[0][i]
        else:
            s += solve(temp) * m[0][i]
    return s


def str2m(string: str) -> Matrix:
    l = list(
        map(lambda x: list(map(lambda y: Expression.from_str(y), x.split())), filter(lambda x: x, string.split("\n"))))
    return Matrix(l)


inp1 = st.text_area(label="矩阵A")
inp2 = st.text_area(label="矩阵B")
option = st.selectbox("运算", ["行列式", "+", "-", "X", "^"])
match option:
    case "行列式":
        for i in [inp1.strip(), inp2.strip()]:
            if not i:
                continue
            m = str2m(i)
            st.markdown(f"$$\n{m.show('v')}=\n$$")
            st.markdown(f"$${m.determinant()}$$")
    case "+" | "-":
        a, b = str2m(inp1.strip()), str2m(inp2.strip())
        if inp1.strip() and inp2.strip():
            try:
                c = a + b if option == "+" else a - b
                st.markdown(f"$$\n{a.show()}{option}{b.show()}={c.show()}\n$$")
                st.markdown(f"##### 计算行列式: $${c.determinant()}$$")
            except AssertionError as e:
                st.error(e)
    case "X":
        if inp1.strip() and inp2.strip():
            a, b = str2m(inp1.strip()), str2m(inp2.strip())
            try:
                c = a * b
                st.markdown(f"$$\n{a.show()}{b.show()}={c.show()}\n$$")
                st.markdown(f"##### 计算行列式: $${c.determinant()}$$")
            except AssertionError as e:
                st.error(e)
            try:
                d = b * a
                st.markdown(f"$$\n{b.show()}{a.show()}={d.show()}\n$$")
                st.markdown(f"##### 计算行列式: $$ {d.determinant()}$$")
            except AssertionError as e:
                st.error(e)

    case "^":
        if inp1.strip() and inp2.strip():
            a, b = str2m(inp1.strip()), int(inp2)
            try:
                if b == -1:
                    c, d = a ** -1
                    st.markdown(f"$$\n{a.show("v")}={c}\n$$")
                    st.markdown(f"$$\n{a.show()}^" + "{*}=" + f"{d.show()}\n$$")
                else:
                    c = a ** b
                    st.markdown(f"$$\n{a.show()}^" + "{" + f"{b}" + "}" + f"={c.show()}\n$$")
                    st.markdown(f"##### 计算行列式: $${c.determinant()}$$")
            except AssertionError as e:
                st.error(e)

st.markdown(r"""使用说明
--
- 计算器支持行列式、矩阵加、减、乘、幂运算，支持部分包含字母的表达式
- 输入矩阵时请用空格将表达式分开，输入为以下形式：
```
a b c  
d e f  
g h i  
```
表示矩阵
$$
\begin{bmatrix}
a&b&c\\
d&e&f\\
g&h&i\\
\end{bmatrix}
$$
- 表达式结构为 **系数+未知量**（允许包含括号嵌套）
- 系数为任意实数
- 未知量为字母+^+幂（可选）
- 例：
```
3aa
5b^-5
34jgc
-114.514ab^67c
78(12a)(34a+b)
```
其值依次为：  
$$3a^{2}$$  
$$5b^{-5}$$  
$$34cgj$$  
$$-114514.ab^{67}c$$  
$$31824a^{2}+936ab$$  
- 加减均为 上$\pm$下
- 纯python项目，作者写的很烂，如有bug欢迎在github上提issue
- ~~或者帮我改代码~~
- [项目地址(github)](https://github.com/jiguancheng/determinant_calculating/)""")
