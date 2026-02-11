window.UICharts = {
  sales(elId, points){
    const el = document.getElementById(elId);
    if(!el || !window.echarts) return;
    const chart = echarts.init(el);
    chart.setOption({
      tooltip:{trigger:'axis'},
      xAxis:{type:'category',data:points.map(p=>p.label)},
      yAxis:{type:'value'},
      series:[{type:'bar',data:points.map(p=>p.sales),itemStyle:{color:'#2563eb'}}]
    });
  }
}
