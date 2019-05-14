import calendar
import datetime


def fst_last_date(month, year=None):
    if not year:
        year = datetime.datetime.now().year
    # 获取当月第一天的星期和当月的总天数
    firstDayWeekDay, monthRange = calendar.monthrange(year, month)
    firstDay = datetime.date(year=year, month=month, day=1)
    lastDay = datetime.date(year=year, month=month, day=monthRange)
    return str(firstDay), str(lastDay)
