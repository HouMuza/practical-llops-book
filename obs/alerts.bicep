param actionGroupId string
param targetResourceId string
param location string = resourceGroup().location
 
resource highLatency 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'llm-p95-latency-high'
  location: 'global'
  properties: {
    enabled: true
    severity: 2
    scopes: [targetResourceId]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'p95Latency'
          metricName: 'RequestLatency'
          operator: 'GreaterThan'
          threshold: 2000
          timeAggregation: 'Average'
          criterionType: 'StaticThresholdCriterion'
        }
      ]
    }
    actions: [{ actionGroupId: actionGroupId }]
  }
}
