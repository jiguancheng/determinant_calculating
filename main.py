import streamlit as st
from typing import List, Dict

alphas = "abcdefghijklmnopqrstuvwxyz"


class Term:
    def __init__(self, coefficient: float | int, variables: Dict[str, int]):
        self.coefficient = coefficient
        self.variables = variables

    def __repr__(self):
        if not any(self.variables.values()):
            return ("+" if self.coefficient >= 0 else "") + str(self.coefficient)
        return f"{'+' if self.coefficient > 0 else ''}{'-' if self.coefficient == -1 else self.coefficient if self.coefficient != 1 else ''}{''.join(map(lambda x: f'{x[0]}^' + '{' + f'{x[1]}' + '}' if x[1] != 1 else str(x[0]), sorted(filter(lambda x: x[1] != 0, self.variables.items()))))}"

    def __eq__(self, other):
        if other == 0:
            return self.coefficient == 0
        if isinstance(other, (int, float)):
            return False if self.coefficient != other or any(self.variables.values()) else True
        if isinstance(other, Term):
            return True if self.coefficient == other.coefficient and self.variables == other.variables else False
        raise TypeError(f"{type(other)} 不能与 Term 比较")

    def __neg__(self):
        return Term(-self.coefficient, self.variables)

    def __copy__(self):
        return Term(self.coefficient, self.variables.copy())


class Expression:
    def __init__(self, terms: List[Term]):
        self.terms = terms
        self.clear()

    def __repr__(self):
        if not any(map(lambda x: x != 0, self.terms)):
            return "$$0$$"
        text = "".join(map(str, self.terms)).lstrip("+")
        return f"$${text}$$"

    def clear(self):
        self.terms = list(filter(lambda x: x != 0, self.terms))

    @staticmethod
    def lstrip(string, char):
        return string if string[0] != char else string[1:]

    @classmethod
    def simple_str(cls, string):
        if string == "":
            return Expression([Term(1, {alphas[_]: 0 for _ in range(26)})])
        term = Expression([Term(1, {alphas[_]: 0 for _ in range(26)})])
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
        string = string.lstrip("+")
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

    def __add__(self, other):
        terms = Expression(self.terms.copy())
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
        raise TypeError(f"{self} 不能与 {other} 运算")

    def __neg__(self):
        terms = list(map(lambda x: -x, self.terms.copy()))
        return Expression(terms)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, Term):
            terms = Expression([i.__copy__() for i in self.terms])
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


def solve(m):
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


def str2m(string: str):
    l = list(map(lambda x: list(map(lambda y: Expression.from_str(y), x.split())), filter(lambda x: x, string.split("\n"))))
    return l


inp = st.text_area(label="input", )
result = st.markdown(str(solve(str2m(inp))) + "\n--" if inp else "")
