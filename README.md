## About

KRRBFNN is implemented Radial basis function network of machine learning.

#### Podfile

```ruby
platform :ios, '8.0'
pod "KRRBFNN", "~> 1.1.0"
```

## How To Get Started

#### Import
``` objective-c
#import "KRRBFNN.h"
KRRBFNN *network = [KRRBFNN sharedNetwork];
```

#### Picking centers by OLS
``` objective-c
// The tolerance is a custom number, in here example is 0.8f.
[network pickCentersByOLSWithTolerance:0.8f];
```

#### Picking centers by Random
``` objective-c
// LimitCount is how many centers do you wanna pick.
[network pickCentersByRandomWithLimitCount:5];
```

#### Random initial weights
``` objective-c
// Random to setup weights of network must after picked centers and added patterns.
[network randomWeightsBetweenMin:-0.25f max:0.25f];
```

#### Training by LMS
``` objective-c
[network trainLMSWithCompletion:^(BOOL success, KRRBFNN *rbfnn, double rmse) {
    NSLog(@"rmse : %f", rmse);
    if( rmse > 0.1f )
    {
        // Save trained parameters of network.
        [rbfnn saveForKey:@"RBFNN_1"];
        // Reset all trained information to prepare next retrain.
        [rbfnn reset];
    }
} eachOutput:^(KRRBFOutputNet *outputNet) {
    NSLog(@"net(%@) the output is %f and target is %f", outputNet.indexKey, outputNet.outputValue, outputNet.targetValue);
}];
```

#### Training by SGA
``` objective-c
//[network pickCentersByOLSWithTolerance:1.0f]; // To use OLS
[network pickCentersByRandomWithLimitCount:5];  // To use Random picking

[network randomWeightsBetweenMin:0.0 max:0.25];

network.learningRate   = 0.8f;
network.toleranceError = 0.001f;
network.maxIteration   = 1000;

[network trainSGAWithCompletion:^(BOOL success, KRRBFNN *rbfnn) {
    NSLog(@"Done in %li the RMSE %f", rbfnn.iterationTimes, rbfnn.rmse);
    [rbfnn saveForKey:@"RBFNN_SGA"];
    [rbfnn reset];
} iteration:^BOOL(NSInteger iteration, double rmse) {
    NSLog(@"Iteration %li the RMSE %f", iteration, rmse);
    // YES means we allow to continue next iteration, NO means don't do next iteration (immediately stop).
    return YES;
}];
```

#### Retrieve saved network information
Retrieving centers and weights.
``` objective-c
// Recover from saved network for key.
[network recoverForKey:@"RBFNN_1"];
```

#### Predication
``` objective-c
// Predicating by trained network.
[network predicateWithPatterns:[self createVerificationPatterns] output:^(NSDictionary<NSString *,NSArray<NSNumber *> *> *outputs) {
    NSLog(@"predicated outputs : %@", outputs);
}];
```

## Version

V1.1.0

## LICENSE

MIT.

