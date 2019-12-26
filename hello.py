# author: xuxiangfeng 
# date: 2019/12/25

class Student:
    __slots__ = ('_name', '_age')

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name


s = Student()
s.name = 34
