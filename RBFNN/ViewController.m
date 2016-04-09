//
//  ViewController.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/23.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "ViewController.h"

#import "KRRBFChoice.h"

#import "KRRBFPattern.h"
#import "KRRBFTarget.h"
#import "KRRBFCenterNet.h"
#import "KRRBFLMS.h"

@interface ViewController ()

@end

@implementation ViewController

#pragma --mark Unit Test
-(void)testOls
{
    NSMutableArray<KRRBFPattern *> *_patterns = [NSMutableArray new];
    
    NSInteger patternItem  = 20; // Testing patterns
    NSInteger featureItem  = 10; // How many features of each pattern
    NSInteger featureScope = 21; // Features of pattern that every scope is start in 0
    NSInteger targetItem   = 5;  // Target outputs of each pattern
    NSInteger targetScope  = 10; // Scope of target outputs that start in 0
    for( NSInteger i=0; i<patternItem; i++ )
    {
        KRRBFPattern *p = [[KRRBFPattern alloc] init];
        p.indexKey      = @(i);
        // 隨機亂數 x 個特徵值其範圍在 0 ~ y 之間
        for( NSInteger f=0; f<featureItem; f++ )
        {
            [p addFeature:( arc4random() % featureScope )];
        }
        
        // Adding the targets of patterns at the same time
        // x output targets of 1 pattern
        for( NSInteger t=0; t<targetItem; t++ )
        {
            [p addTarget:( arc4random() % targetScope )];
        }
        
        [_patterns addObject:p];
        
        //NSLog(@"features : %@", p.features);
        //NSLog(@"targets : %@", p.targets);
        //NSLog(@"========== \n\n");
        
    }
    
    // 製作 RBFTarget
    // 設定 sameSequences : 先 Loop 所有的 Patterns Outputs，之後取出同維度的所有 Target Output 集合在一起
    // 先取出有幾個期望輸出 (幾個分類)
    NSInteger targetCount = [[[_patterns firstObject] targets] count];
    // 再依序 Loop 所有的 Patterns 將它們的期望輸出值依序放入各自同維度的陣列裡集合起來
    NSMutableArray<KRRBFTarget *> *_targets = [NSMutableArray new];
    for( NSInteger i=0; i<targetCount; i++ )
    {
        KRRBFTarget *targetOutput = [[KRRBFTarget alloc] init];
        for( KRRBFPattern *p in _patterns )
        {
            [targetOutput.sameSequences addObject:[p.targets objectAtIndex:i]];
        }
        [_targets addObject:targetOutput];
        
        //NSLog(@"targetOutput.sameSequences : %@", targetOutput.sameSequences);
    }
    
    //NSLog(@"_targets : %@", _targets);
    
    KRRBFOLS *ols = [KRRBFChoice sharedChoice].ols;
    ols.tolerance = 0.8f; // = 一般 NN 的 0.2 收斂誤差
    
    // OLS 選取中心點
    NSArray <KRRBFCenterNet *> *choseCenters = [ols chooseWithPatterns:_patterns targets:_targets];
    
    NSArray *_newWeights = [[KRRBFLMS sharedLMS] weightsWithCenters:choseCenters patterns:_patterns targets:_targets];
    
    NSLog(@"_newWeights : %@", _newWeights);
   
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
