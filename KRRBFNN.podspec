Pod::Spec.new do |s|
  s.name         = "KRRBFNN"
  s.version      = "1.1.0"
  s.summary      = "KRRBFNN is implemented Radial basis function network of machine learning."
  s.description  = <<-DESC
                   KRRBFNN is a Radial basis function network used Guassian function, implemented OLS, LMS, SGA, Random algorithms.
                   DESC
  s.homepage     = "https://github.com/Kalvar/ios-KRRBFNN"
  s.license      = { :type => 'MIT', :file => 'LICENSE' }
  s.author       = { "Kalvar Lin" => "ilovekalvar@gmail.com" }
  s.social_media_url = "https://twitter.com/ilovekalvar"
  s.source       = { :git => "https://github.com/Kalvar/ios-KRRBFNN.git", :tag => s.version.to_s }
  s.platform     = :ios, '8.0'
  s.requires_arc = true
  s.public_header_files = 'ML/**/*.h'
  s.source_files = 'ML/**/*.{h,m}'
  s.frameworks   = 'Accelerate', 'Foundation'
end 