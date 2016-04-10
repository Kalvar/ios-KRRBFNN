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

@end

@implementation ViewController

#pragma --mark Unit Test
-(void)testOls
{
    KRRBFNN *network           = [KRRBFNN sharedNetwork];
    //rbfnn.centerChoiceMethod = KRRBFNNCenterChoiceOLS;
    //rbfnn.learningMethod     = KRRBFNNLearningLMS;
    
    // To add patterns (KRRBFPattern)
    NSInteger patternItem  = 20; // Testing patterns
    NSInteger featureItem  = 10; // How many features of each pattern
    NSInteger featureScope = 21; // Features of pattern that every scope is start in 0
    NSInteger targetItem   = 5;  // Target outputs of each pattern
    NSInteger targetScope  = 10; // Scope of target outputs that start in 0
    for( NSInteger i=0; i<patternItem; i++ )
    {
        KRRBFPattern *p = [[KRRBFPattern alloc] init];
        
        // Setup that index
        p.indexKey      = @(i);
        
        // Setup that features of pattern
        for( NSInteger f=0; f<featureItem; f++ )
        {
            [p addFeature:( arc4random() % featureScope )];
        }
        
        // Setup that target-outputs of pattern (多輸出)
        for( NSInteger t=0; t<targetItem; t++ )
        {
            [p addTarget:( arc4random() % targetScope )];
        }
        
        [network addPattern:p];
    }
    
    [network pickingCentersByOLSWithTolerance:0.8f setToCenters:YES];
    [network trainingByLMSWithCompletion:^(BOOL success, KRRBFNN *rbfnn) {
        
    }];
    
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    [self testOls];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
