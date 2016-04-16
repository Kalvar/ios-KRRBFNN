//
//  ViewController.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/23.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "ViewController.h"
#import "KRRBFNN.h"

@interface ViewController ()

@property (nonatomic, assign) NSInteger patternItem;
@property (nonatomic, assign) NSInteger featureItem;
@property (nonatomic, assign) NSInteger featureScope;
@property (nonatomic, assign) NSInteger targetItem;
@property (nonatomic, assign) NSInteger targetScope;

@end

@implementation ViewController

#pragma --mark Examples
-(void)setupParameters
{
    _patternItem  = 20; // Testing patterns
    _featureItem  = 10; // How many features of each pattern
    _featureScope = 21; // Features of pattern that every scope is start in 0
    _targetItem   = 5;  // Target outputs of each pattern
    _targetScope  = 10; // Scope of target outputs that start in 0
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
            [p addFeature:( arc4random() % _featureScope )];
        }
        
        // Setup that target-outputs of pattern (多輸出)
        for( NSInteger t=0; t<_targetItem; t++ )
        {
            [p addTarget:( arc4random() % _targetScope )];
        }
        
        //NSLog(@"p(%@).targets : %@", p.indexKey, p.targets);
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
            [p addFeature:( arc4random() % _featureScope )];
        }
        //NSLog(@"p(%@).features : %@", p.indexKey, p.features);
        [_verifications addObject:p];
    }
    return _verifications;
}

-(void)testOls
{
    __weak typeof(self) _weakSelf = self;
    
    KRRBFNN *network = [[KRRBFNN alloc] init];
    [network addPatterns:[self createTrainingPatterns]];
    [network pickCentersByOLSWithTolerance:0.8f];
    [network trainLMSWithCompletion:^(BOOL success, KRRBFNN *rbfnn, double rmse) {
        NSLog(@"rmse : %f", rmse);
        if( rmse > 0.1f )
        {
            // Save that trained parameters of network.
            [rbfnn saveForKey:@"RBFNN_1"];
            // Reset all trained information to prepare next retrain.
            [rbfnn reset];
            // Recover from saved network for key.
            [rbfnn recoverForKey:@"RBFNN_1"];
            
            __strong typeof(self) _strongSelf = _weakSelf;
            // Predicating by trained network.
            [rbfnn predicateWithPatterns:[_strongSelf createVerificationPatterns] output:^(NSDictionary<NSString *,NSArray<NSNumber *> *> *outputs) {
                NSLog(@"predicated outputs : %@", outputs);
            }];
        }
    } eachOutput:^(KRRBFOutputNet *outputNet) {
        //NSLog(@"net(%@) the output is %f and target is %f", outputNet.indexKey, outputNet.outputValue, outputNet.targetValue);
    }];
    
}

-(void)testRandom
{
    __weak typeof(self) _weakSelf = self;
    
    KRRBFNN *network = [[KRRBFNN alloc] init];
    [network addPatterns:[self createTrainingPatterns]];
    [network pickCentersByRandomWithLimitCount:5];
    [network trainLMSWithCompletion:^(BOOL success, KRRBFNN *rbfnn, double rmse) {
        NSLog(@"rmse : %f", rmse);
        if( rmse > 0.1f )
        {
            [rbfnn saveForKey:@"RBFNN_2"];
            [rbfnn reset];
            [rbfnn recoverForKey:@"RBFNN_2"];
            
            __strong typeof(self) _strongSelf = _weakSelf;
            // Predicating by trained network.
            [rbfnn predicateWithPatterns:[_strongSelf createVerificationPatterns] output:^(NSDictionary<NSString *,NSArray<NSNumber *> *> *outputs) {
                NSLog(@"predicated outputs : %@", outputs);
            }];
        }
    } eachOutput:^(KRRBFOutputNet *outputNet) {
        //NSLog(@"net(%@) the output is %f and target is %f", outputNet.indexKey, outputNet.outputValue, outputNet.targetValue);
    }];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    [self setupParameters];
    [self testOls];
    [self testRandom];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
