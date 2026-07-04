param deploymentResourceId string
param location string = resourceGroup().location
 
resource autoscale 'Microsoft.Insights/autoscaleSettings@2022-10-01' = {
  name: 'qwen3-autoscale'
  location: location
  properties: {
    enabled: true
    targetResourceUri: deploymentResourceId
    profiles: [
      {
        name: 'default'
        capacity: { minimum: '1', maximum: '10', default: '2' }
        rules: [
          {
            metricTrigger: {
              metricName: 'GpuUtilizationPercentage'
              metricResourceUri: deploymentResourceId
              timeGrain: 'PT1M'
              statistic: 'Average'
              timeWindow: 'PT5M'
              timeAggregation: 'Average'
              operator: 'GreaterThan'
              threshold: 75
            }
            scaleAction: { direction: 'Increase', type: 'ChangeCount', value: '1', cooldown: 'PT5M' }
          }
          {
            metricTrigger: {
              metricName: 'GpuUtilizationPercentage'
              metricResourceUri: deploymentResourceId
              timeGrain: 'PT1M'
              statistic: 'Average'
              timeWindow: 'PT10M'
              timeAggregation: 'Average'
              operator: 'LessThan'
              threshold: 25
            }
            scaleAction: { direction: 'Decrease', type: 'ChangeCount', value: '1', cooldown: 'PT10M' }
          }
        ]
      }
    ]
  }
}
