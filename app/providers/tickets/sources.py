"""景点名称到官方票务页的映射；这里只保存网址，不保存票价。"""

OFFICIAL_TICKET_SOURCES = {
    "颐和园": ("颐和园官网·票务服务", "https://www.summerpalace.net.cn/visit.html", "summer_palace"),
    "颐和园博物馆": ("颐和园官网·票务服务", "https://www.summerpalace.net.cn/visit.html", "summer_palace_museum"),
    "颐和园文昌院": ("颐和园官网·票务服务", "https://www.summerpalace.net.cn/visit.html", "summer_palace_museum"),
    "文昌院": ("颐和园官网·票务服务", "https://www.summerpalace.net.cn/visit.html", "summer_palace_museum"),
    "故宫": ("故宫博物院·票务政策", "https://www.dpm.org.cn/singles_detail/257830.html", "palace_museum"),
    "故宫博物院": ("故宫博物院·票务政策", "https://www.dpm.org.cn/singles_detail/257830.html", "palace_museum"),
    "景山": ("北京旅游网·景山公园", "https://s.visitbeijing.com.cn/attraction/117811", "jingshan"),
    "景山公园": ("北京旅游网·景山公园", "https://s.visitbeijing.com.cn/attraction/117811", "jingshan"),
    "紫竹院": ("北京旅游网·紫竹院公园", "https://s.visitbeijing.com.cn/attraction/101455", "free_page"),
    "紫竹院公园": ("北京旅游网·紫竹院公园", "https://s.visitbeijing.com.cn/attraction/101455", "free_page"),
    "什刹海": ("北京旅游网·什刹海风景区", "https://s.visitbeijing.com.cn/attraction/117800", "free_page"),
    "什刹海公园": ("北京旅游网·什刹海风景区", "https://s.visitbeijing.com.cn/attraction/117800", "free_page"),
    "什刹海风景区": ("北京旅游网·什刹海风景区", "https://s.visitbeijing.com.cn/attraction/117800", "free_page"),
    "什刹海旅游风景区": ("北京旅游网·什刹海风景区", "https://s.visitbeijing.com.cn/attraction/117800", "free_page"),
    "天坛": ("天坛公园官方网站", "https://www.tiantanpark.cn/index.html", "tiantan"),
    "天坛公园": ("天坛公园官方网站", "https://www.tiantanpark.cn/index.html", "tiantan"),
    "恭王府": ("恭王府博物馆官方网站", "https://www.gongwangfu.com/", "gongwangfu"),
    "恭王府博物馆": ("恭王府博物馆官方网站", "https://www.gongwangfu.com/", "gongwangfu"),
    "南锣鼓巷": ("北京旅游网·南锣鼓巷", "https://s.visitbeijing.com.cn/attraction/117821", "free_page"),
}
