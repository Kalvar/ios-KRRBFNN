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
    ols.tolerance = 0.999f; // = 一般 NN 的 0.001 收斂誤差
    
    // OLS 選取中心點
    NSArray <KRRBFCenterNet *> *choseCenters = [ols chooseWithPatterns:_patterns targets:_targets];
    
    NSArray *_newWeights = [[KRRBFLMS sharedLMS] weightsWithCenters:choseCenters patterns:_patterns targets:_targets];
    
    NSLog(@"_newWeights : %@", _newWeights);
    
   /*
# 有幾個實作想法 :
    1. 把 2 個權重修正方法分開寫 :
    - a). 在 OLS 這支 class 裡寫「最小平方法 (LMS)」求權重
    - b). 另外再寫一支 SGA 來修正權重
    2. 有幾種用法 :
    - a). OLS 選中心點 -> 用 LMS 解聯立直接求出權重結果 -> 再用 SGA 來做後續修正提昇精度
    - b). OLS 選中心點 -> 亂數給權重 (-0.25 ~ 0.25) -> 再用 SGA 來做修正
    - c). Random 選中心點 -> 用 LMS 解聯立直接求出權重結果 -> 再用 SGA 來做後續修正提昇精度
    - d). Random 選中心點 -> 亂數給權重 (-0.25 ~ 0.25) -> 再用 SGA 來做修正
    
    05/04/2016 已決定以 a ~ d 的方法來實作。
    
    
    // LMS 求權重
    KRRBFLMS *lms    = [KRRBFLMS sharedLMS];
    NSArray *weights = [lms weightsWithCenters:choseCenters];
    
    
    
# Continue testing steps :
    
    1). 實作 5.4.1 的「利用最小平方法求得權重向量」 (用最小平方法求權重)
    
    NSDictionary *_results = [self _calculateDistancesFromCenters:_choseCenters];
    NSNumber *_maxDistance = [self _getMaxDistanceByResults:_results];
    double _sigma          = [_maxDistance doubleValue] / sqrt([_patterns count]);
    
    NSLog(@"_sigma : %lf", _sigma);
    
    在這裡跑 P.186 先算所有 Patterns 到 Chose Centers 的 Phi (RBF 值)後，再一次解聯立方程式求出最初始要設定的 Weights*
    
    之後再用 Weights 來計算網路推估值
    
    最後再算這次的 RMSE，即完成 OLS 的完整階段算法
    
    
    
    2). 之後再試驗 5.4.2 的 SGA 方法來修重各個參數的實作
    
    SGA 會修 3 個參數，而初始的所有參數值都亂數給即可 ( -0.25 ~ 0.25 )
    
    */
    
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
