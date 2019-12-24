# author: xuxiangfeng 
# date: 2019/12/25

class Student:

    @property
    def score(self):
        return self._value

    @score.setter
    def score(self, value):
        self._value = value


s = Student()

s.score = 34

print(s.score)