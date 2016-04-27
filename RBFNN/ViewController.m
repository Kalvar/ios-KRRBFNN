//
//  ViewController.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/23.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "ViewController.h"
#import "KRRBFNN.h"
#import "KRMathLib.h"

@interface ViewController ()

@property (nonatomic, assign) NSInteger patternItem;
@property (nonatomic, assign) NSInteger featureItem;
@property (nonatomic, assign) NSInteger featureScope;
@property (nonatomic, assign) NSInteger targetItem;
@property (nonatomic, assign) NSInteger targetScope;

@property (nonatomic, assign) BOOL featureScaling;

@end

@implementation ViewController

#pragma --mark Examples
-(void)setupParameters
{
    _patternItem    = 20; // Testing patterns
    _featureItem    = 10; // How many features of each pattern
    _featureScope   = 21; // Features of pattern that every scope is start in 0
    
    _targetItem     = 5;  // Target outputs of each pattern
    _targetScope    = 10; // Scope of target outputs that start in 0
    
    _featureScaling = NO; // Does start feature scaling ? It it YES, the training patterns will ignore _featureScope & _targetScope.
}

-(NSArray <KRRBFPattern *> *)createTrainingPatterns
{
    NSMutableArray *patterns = [NSMutableArray new];
    // Creates training patterns (KRRBFPattern)
    for( NSInteger i=0; i<_patternItem; i++ )
    {
        KRRBFPattern *p = [[KRRBFPattern alloc] init];
        
        // Setup that index
        p.indexKey      = @(i);
        
        // Setup that features of pattern
        for( NSInteger f=0; f<_featureItem; f++ )
        {
            if( _featureScaling )
            {
                [p addFeature:[[KRMathLib sharedLib] randomDoubleMax:1.0f min:-1.0f]];
            }
            else
            {
                [p addFeature:( arc4random() % _featureScope )];
            }
        }
        
        // Setup that target-outputs of pattern (多輸出)
        for( NSInteger t=0; t<_targetItem; t++ )
        {
            if( _featureScaling )
            {
                [p addTarget:[[KRMathLib sharedLib] randomDoubleMax:1.0f min:0.0f]];
            }
            else
            {
                [p addTarget:( arc4random() % _targetScope )];
            }
        }
        
        //NSLog(@"train p(%@).features : %@", p.indexKey, p.features);
        NSLog(@"train p(%@).targets : %@", p.indexKey, p.targets);
        [patterns addObject:p];
    }
    return patterns;
}

-(NSArray <KRRBFPattern *> *)createVerificationPatterns
{
    // Creates verification patterns
    NSMutableArray *_verifications = [NSMutableArray new];
    for( NSInteger i=0; i<_patternItem/2; i++ )
    {
        KRRBFPattern *p = [[KRRBFPattern alloc] init];
        p.indexKey      = @(i);
        for( NSInteger f=0; f<_featureItem; f++ )
        {
            if( _featureScaling )
            {
                [p addFeature:[[KRMathLib sharedLib] randomDoubleMax:1.0f min:-1.0f]];
            }
            else
            {
                [p addFeature:( arc4random() % _featureScope )];
            }
        }
        //NSLog(@"verify p(%@).features : %@", p.indexKey, p.features);
        [_verifications addObject:p];
    }
    return _verifications;
}

-(void)testOLS
{
    __weak typeof(self) _weakSelf = self;
    
    KRRBFNN *network = [[KRRBFNN alloc] init];
    [network addPatterns:[self createTrainingPatterns]];
    [network pickCentersByOLSWithTolerance:0.8f];
    [network trainLMSWithCompletion:^(BOOL success, KRRBFNN *rbfnn) {
        double rmse = rbfnn.rmse;
        NSLog(@"rmse : %f", rmse);
        if( rmse > 0.1f )
        {
            // Save that trained parameters of network.
            [rbfnn saveForKey:@"RBFNN_OLS"];
            // Reset all trained information to prepare next retrain.
            [rbfnn reset];
            // Recover from saved network for key.
            [rbfnn recoverForKey:@"RBFNN_OLS"];
            
            __strong typeof(self) _strongSelf = _weakSelf;
            // Predicating by trained network.
            [rbfnn predicateWithPatterns:[_strongSelf createVerificationPatterns] output:^(NSDictionary<NSString *,NSArray<NSNumber *> *> *outputs) {
                NSLog(@"predicated outputs : %@", outputs);
            }];
        }
    } patternOutput:^(NSArray<KRRBFOutputNet *> *patternOutputs) {
        // That outpus of each pattern.
        for( KRRBFOutputNet *outputNet in patternOutputs )
        {
            NSLog(@"net(%@) the output is %f and target is %f", outputNet.indexKey, outputNet.outputValue, outputNet.targetValue);
        }
    }];
}

-(void)testRandom
{
    __weak typeof(self) _weakSelf = self;
    
    KRRBFNN *network = [[KRRBFNN alloc] init];
    [network addPatterns:[self createTrainingPatterns]];
    [network pickCentersByRandomWithLimitCount:5];
    [network trainLMSWithCompletion:^(BOOL success, KRRBFNN *rbfnn) {
        double rmse = rbfnn.rmse;
        NSLog(@"rmse : %f", rmse);
        if( rmse > 0.1f )
        {
            [rbfnn saveForKey:@"RBFNN_Random"];
            [rbfnn reset];
            [rbfnn recoverForKey:@"RBFNN_Random"];
            
            __strong typeof(self) _strongSelf = _weakSelf;
            // Predicating by trained network.
            [rbfnn predicateWithPatterns:[_strongSelf createVerificationPatterns] output:^(NSDictionary<NSString *,NSArray<NSNumber *> *> *outputs) {
                NSLog(@"predicated outputs : %@", outputs);
            }];
        }
    } patternOutput:^(NSArray<KRRBFOutputNet *> *patternOutputs) {
        for( KRRBFOutputNet *outputNet in patternOutputs )
        {
            NSLog(@"net(%@) the output is %f and target is %f", outputNet.indexKey, outputNet.outputValue, outputNet.targetValue);
        }
    }];
}

-(void)testSGA
{
    __weak typeof(self) _weakSelf = self;
    
    KRRBFNN *network = [[KRRBFNN alloc] init];
    [network addPatterns:[self createTrainingPatterns]];
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
        [rbfnn recoverForKey:@"RBFNN_SGA"];
        
        __strong typeof(self) _strongSelf = _weakSelf;
        // Predicating by trained network.
        [rbfnn predicateWithPatterns:[_strongSelf createVerificationPatterns] output:^(NSDictionary<NSString *,NSArray<NSNumber *> *> *outputs) {
            NSLog(@"predicated outputs : %@", outputs);
        }];
    } iteration:^BOOL(NSInteger iteration, double rmse) {
        NSLog(@"Iteration %li the RMSE %f", iteration, rmse);
        // YES means we allow to continue next iteration, NO means don't do next iteration (immediately stop).
        return YES;
    }];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    [self setupParameters];
    [self testOLS];
    [self testRandom];
    [self testSGA];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
