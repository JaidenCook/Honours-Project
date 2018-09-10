#Script for parallel processing.

cpu=10
n=1
nthread=cpu*n

require 'peach'
puts "starting"
puts ARGV[0]

i=0
File.readlines(ARGV[0]).peach(nthread) do |l|
  i+=1
  puts i
  system(l)

end

system('say "script finished"')

